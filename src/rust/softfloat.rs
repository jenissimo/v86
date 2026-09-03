// Native f64 FPU implementation - replaces Berkeley SoftFloat
// Trades 80-bit extended precision for 64-bit double precision (acceptable for games).

/// x87 rounding control and precision control are read from the LIVE control word, never
/// cached. A derived copy is only as fresh as its last sync, and three paths write
/// fpu_control_word without going through set_control_word: the host's per-thread x87
/// restore (fpu-helper.ts writes the 134-byte snapshot straight into wasm memory), CPU
/// reset, and any state-region restore. A cached mode therefore belongs to whichever
/// thread last executed FLDCW, while the JIT's inline FIST reads the control word itself
/// (codegen.rs gen_fpu_round_f64_bits_to_i32) — so the two implementations of the same
/// instruction disagree, and a guest CRT floor/ceil (FLDCW rc; FRNDINT; FLDCW restore)
/// preempted mid-sequence rounds by another thread's mode.
#[inline]
fn control_word() -> u16 { unsafe { *crate::cpu::global_pointers::fpu_control_word } }

/// RC field, in the x87 encoding the hardware uses: 0=NearEven 1=Down 2=Up 3=Trunc.
/// Same field, same encoding the JIT's inline path decodes.
#[inline]
fn rounding_mode() -> u16 { control_word() >> 10 & 3 }

#[inline]
fn round_by_rc(f: f64) -> f64 {
    match rounding_mode() {
        1 => f.floor(),
        2 => f.ceil(),
        3 => f.trunc(),
        _ => round_ties_even(f),
    }
}

/// PC=00 (24-bit single) rounds every arithmetic result to f32. F80 is f64-backed so
/// P64/P80 already match (53-bit); only P32 needs the extra round.
///
/// Relaxed mode honours PC too — it is the one x87 control it cannot spend. A title that
/// compares a freshly computed value against one it stored as f32 never sees them equal
/// once the result keeps f64 mantissa bits, which corrupts the title's own data rather
/// than merely costing it precision. What must NOT happen is gating the CODEGEN fast path
/// on PC: that sent every fadd/fmul of a PC=24 title (LS3D sets PC=24 once at init)
/// through this helper, 19.6% of Mafia's busy time. Nor specializing the compiled block on
/// PC — the CRT saves and restores the control word around every transcendental, so the
/// class flips twice per call and the invalidation empties the JIT cache continuously
/// (measured: 51.7 -> 3.6 FPS). The inline path reads the live control word and rounds
/// branchlessly instead; see codegen's gen_fpu_round_f64_by_precision_control.
#[inline]
fn apply_precision(f: f64) -> f64 {
    if control_word() >> 8 & 3 == 0 { f as f32 as f64 } else { f }
}

/// Relaxed FPU mode: store raw f64 bits directly in F80.mantissa with RELAXED_TAG.
/// Eliminates the F80 biasing overhead in to_f64()/of_f64() (~8% WASM CPU for Re-Volt).
/// Safe for games: precision difference between f80 and f64 is imperceptible.
/// Enable via set_relaxed_fpu(1) WASM export at emulator startup.
static mut FPU_RELAXED: bool = false;

/// Magic sign_exponent value used to tag relaxed-format F80 values.
/// NOT an impossible encoding: 0x7FFE is the exponent field of every finite 2^16383-scale
/// x87 value (LDBL_MAX among them), so it is only honoured while relaxed mode is on and
/// fpu_load_m80 rewrites the one guest-supplied image that could alias it.
pub const RELAXED_TAG: u16 = 0x7FFE;

#[no_mangle]
pub extern "C" fn set_relaxed_fpu(enabled: u32) {
    let on = enabled != 0;
    unsafe {
        if on == FPU_RELAXED {
            return;
        }
        // Live registers are read under the new mode's tag rules, so resolve them across
        // the switch — in BOTH directions, and always decided before the flip, since
        // afterwards the two forms are indistinguishable:
        //   relaxed -> strict: f64 bits would be decoded as a true 80-bit encoding.
        //   strict -> relaxed: a genuine 2^16383-scale value (exponent == RELAXED_TAG,
        //     LDBL_MAX among them) would be misread as f64 bits — the same alias
        //     fpu_load_m80 rewrites, re-expressed the same way.
        let mut aliased = [0u64; 8];
        let mut alias_mask = 0u8;
        for i in 0..8 {
            let p = crate::cpu::global_pointers::fpu_st.offset(i);
            if on {
                if (*p).sign_exponent == RELAXED_TAG {
                    aliased[i as usize] = (*p).to_f64_strict();
                    alias_mask |= 1 << i;
                }
            }
            else {
                *p = (*p).to_true_f80();
            }
        }
        crate::cpu::cpu::mark_fpu_simd_dirty();
        FPU_RELAXED = on;
        for i in 0..8 {
            if alias_mask >> i & 1 != 0 {
                *crate::cpu::global_pointers::fpu_st.offset(i) =
                    F80::of_f64(aliased[i as usize]);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn get_relaxed_fpu() -> u32 {
    unsafe { FPU_RELAXED as u32 }
}

#[allow(dead_code)]
pub fn is_fpu_relaxed() -> bool {
    unsafe { FPU_RELAXED }
}

/// When off (default), the relaxed fast path emits no hit/fallback counter increment.
/// Toggle on only to measure hit-rate: it is a CODEGEN input, so the flip clears the JIT
/// cache itself (already-compiled blocks would otherwise stay silent) and participates in
/// jit_codegen_fingerprint, so an AOT cache cannot bind counter-bearing blocks to a run
/// that asked for none.
static mut FPU_RELAXED_STATS: bool = false;

#[no_mangle]
pub extern "C" fn set_fpu_relaxed_stats(enabled: u32) {
    unsafe {
        if FPU_RELAXED_STATS == (enabled != 0) {
            return;
        }
        FPU_RELAXED_STATS = enabled != 0;
    }
    crate::jit::jit_clear_cache_js();
}

#[no_mangle]
pub extern "C" fn get_fpu_relaxed_stats() -> u32 {
    unsafe { FPU_RELAXED_STATS as u32 }
}

#[allow(dead_code)]
pub fn is_fpu_relaxed_stats() -> bool {
    unsafe { FPU_RELAXED_STATS }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct F80 {
    pub mantissa: u64,
    pub sign_exponent: u16,
}
impl F80 {
    pub const ZERO: F80 = F80 {
        mantissa: 0,
        sign_exponent: 0,
    };
    pub const ONE: F80 = F80 {
        mantissa: 0x8000000000000000,
        sign_exponent: 0x3FFF,
    };
    pub const LN_10: F80 = F80 {
        mantissa: 0x935D8DDDAAA8B000,
        sign_exponent: 0x4000,
    };
    pub const LN_2: F80 = F80 {
        mantissa: 0xB17217F7D1CF7800,
        sign_exponent: 0x3FFE,
    };
    pub const PI: F80 = F80 {
        mantissa: 0xC90FDAA22168C000,
        sign_exponent: 0x4000,
    };
    pub const LOG2_E: F80 = F80 {
        mantissa: 0xB8AA3B295C17F000,
        sign_exponent: 0x3FFF,
    };
    pub const INDEFINITE_NAN: F80 = F80 {
        mantissa: 0xC000000000000000,
        sign_exponent: 0x7FFF,
    };
    pub const POS_INFINITY: F80 = F80 {
        mantissa: 0x8000000000000000,
        sign_exponent: 0x7FFF,
    };
    pub const NEG_INFINITY: F80 = F80 {
        mantissa: 0x8000000000000000,
        sign_exponent: 0xFFFF,
    };

    /// RELAXED_TAG is a real exponent field (2^16383), so it only means "mantissa holds
    /// raw f64 bits" while the mode that writes it is on. Every tag test goes through here.
    #[inline]
    fn is_relaxed(&self) -> bool {
        unsafe { FPU_RELAXED && self.sign_exponent == RELAXED_TAG }
    }
    #[inline]
    fn both_relaxed(&self, other: &F80) -> bool {
        unsafe {
            FPU_RELAXED
                && self.sign_exponent == RELAXED_TAG
                && other.sign_exponent == RELAXED_TAG
        }
    }

    /// Relaxed values carry their sign in bit 63 of the f64 bits, not in
    /// `sign_exponent` — which is the constant RELAXED_TAG, so reading bit 15
    /// reports every value as positive.
    pub fn sign(&self) -> bool {
        if self.is_relaxed() { return self.mantissa >> 63 == 1; }
        (self.sign_exponent >> 15) == 1
    }
    pub fn exponent(&self) -> i16 {
        let v = self.to_true_f80();
        (v.sign_exponent as i16 & 0x7FFF) - 0x3FFF
    }

    pub fn to_f64(&self) -> u64 {
        // Relaxed fast path: f64 bits stored directly, skip biasing
        if self.is_relaxed() {
            return self.mantissa;
        }
        self.to_f64_strict()
    }

    // Decode as a true 80-bit value, ignoring relaxed mode (needed where a raw guest image
    // must be read even though its exponent field can equal RELAXED_TAG).
    pub fn to_f64_strict(&self) -> u64 {
        let sign = (self.sign_exponent >> 15) as u64;
        let exp = (self.sign_exponent & 0x7FFF) as i32;
        let mant = self.mantissa;

        // Zero (positive or negative)
        if exp == 0 && mant == 0 {
            return sign << 63;
        }

        // NaN or Infinity
        if exp == 0x7FFF {
            if mant == 0x8000000000000000 {
                // Infinity
                return (sign << 63) | (0x7FFu64 << 52);
            }
            // NaN - preserve as much payload as possible
            let payload = (mant & 0x3FFFFFFFFFFFFFFF) >> 11;
            let quiet = (mant >> 62) & 1;
            return (sign << 63) | (0x7FFu64 << 52) | (quiet << 51) | (payload & 0x7FFFFFFFFFFFF);
        }

        // Denormal F80 (exponent == 0, mantissa != 0)
        if exp == 0 {
            // F80 pseudo-denormals: exponent 0 with integer bit set
            // These are equivalent to exponent 1 in F80
            // Extremely small - will underflow to zero in f64
            return sign << 63;
        }

        // Normal: rebias exponent from F80 (bias 16383) to f64 (bias 1023)
        let f64_exp = exp - 16383 + 1023;

        if f64_exp >= 0x7FF {
            // Overflow -> infinity
            return (sign << 63) | (0x7FFu64 << 52);
        }

        if f64_exp <= 0 {
            // Subnormal f64 or underflow
            let shift = 1 - f64_exp;
            if shift >= 64 {
                return sign << 63; // underflow to zero
            }
            // Strip the explicit J-bit and shift mantissa for f64 subnormal
            let f64_mant = mant >> (11 + shift as u32);
            return (sign << 63) | f64_mant;
        }

        // Normal case: strip explicit J-bit (bit 63), take top 52 bits of remaining 63
        let f64_mant = (mant & 0x7FFFFFFFFFFFFFFF) >> 11;
        (sign << 63) | ((f64_exp as u64) << 52) | f64_mant
    }

    pub fn of_f64(src: u64) -> F80 {
        // Relaxed fast path: store f64 bits directly, skip biasing
        unsafe {
            if FPU_RELAXED {
                return F80 { mantissa: src, sign_exponent: RELAXED_TAG };
            }
        }
        F80::of_f64_strict(src)
    }

    // Bit-exact f64 -> true 80-bit, ignoring relaxed mode (needed when a real
    // exponent field must be produced, e.g. FSCALE / store to m80).
    pub fn of_f64_strict(src: u64) -> F80 {
        let sign = (src >> 63) as u16;
        let exp = ((src >> 52) & 0x7FF) as i32;
        let mant = src & 0xFFFFFFFFFFFFF;

        // Zero
        if exp == 0 && mant == 0 {
            return F80 {
                mantissa: 0,
                sign_exponent: sign << 15,
            };
        }

        // NaN or Infinity
        if exp == 0x7FF {
            if mant == 0 {
                // Infinity
                return F80 {
                    mantissa: 0x8000000000000000,
                    sign_exponent: (sign << 15) | 0x7FFF,
                };
            }
            // NaN - reconstruct F80 NaN
            let quiet = (mant >> 51) & 1;
            let payload = (mant & 0x7FFFFFFFFFFFF) << 11;
            return F80 {
                mantissa: 0x8000000000000000 | (quiet << 62) | payload,
                sign_exponent: (sign << 15) | 0x7FFF,
            };
        }

        // Denormal f64
        if exp == 0 {
            // Normalize: find the leading 1 bit
            let shift = mant.leading_zeros() - 12; // 12 because top 12 bits of u64 are unused
            let normalized_mant = mant << (shift + 1); // shift out the leading 1, then we add J-bit
            // A subnormal is mant * 2^-1074 with its leading 1 at bit L = 51 - shift, so the
            // f80 field is 16383 + L - 1074 = 16383 - 1023 - shift. Biasing it as if the
            // implicit bit were present (1 - 1023) doubles the value.
            let f80_exp = (-1023 + 16383 - shift as i32) as u16;
            return F80 {
                mantissa: 0x8000000000000000 | (normalized_mant << 11),
                sign_exponent: (sign << 15) | f80_exp,
            };
        }

        // Normal: rebias exponent from f64 (bias 1023) to F80 (bias 16383)
        let f80_exp = (exp - 1023 + 16383) as u16;
        // Set explicit J-bit (integer bit) and expand mantissa from 52 to 63 bits
        let f80_mant = 0x8000000000000000 | (mant << 11);
        F80 {
            mantissa: f80_mant,
            sign_exponent: (sign << 15) | f80_exp,
        }
    }

    fn of_f64x(src: f64) -> F80 { F80::of_f64(f64::to_bits(src)) }
    fn to_f64x(&self) -> f64 { f64::from_bits(self.to_f64()) }

    pub fn of_f32(src: i32) -> F80 {
        let f = f32::from_bits(src as u32);
        F80::of_f64((f as f64).to_bits())
    }

    pub fn to_f32(&self) -> i32 {
        let f = f64::from_bits(self.to_f64());
        (f as f32).to_bits() as i32
    }

    pub fn of_i32(src: i32) -> F80 {
        F80::of_f64((src as f64).to_bits())
    }

    /// `fild m64` is EXACT on real x87 — the 80-bit format carries a 64-bit mantissa, so
    /// every i64 lands without rounding. Going through f64 (53-bit) rounds away the low
    /// bits of any |value| >= 2^53, and `fild qword`/`fistp qword` is a 90s CRT block-COPY
    /// idiom, so that rounding corrupts COPIED DATA — one wrong 16-bit word per 8 bytes.
    /// Build the true 80-bit encoding instead. Relaxed mode cannot hold this exactly
    /// either, so an integer load is always strict; a mixed pair already falls back to the
    /// strict arithmetic path, and an i64's exponent (<= 0x403E) can never alias
    /// RELAXED_TAG.
    pub fn of_i64(src: i64) -> F80 {
        if src == 0 {
            return F80::ZERO;
        }
        let sign = if src < 0 { 1u16 } else { 0 };
        let mag = (src as i128).unsigned_abs() as u64;
        let shift = mag.leading_zeros() as u16;
        F80 {
            mantissa: mag << shift,
            sign_exponent: (sign << 15) | (0x3FFF + 63 - shift),
        }
    }

    /// Exact i64 for a true 80-bit value — the inverse of `of_i64`, and the reason a
    /// fild/fistp round trip is lossless. `None` means "the f64 path is correct here or
    /// the value is out of range": a relaxed encoding, NaN/Inf, an unnormal, |x| < 1
    /// (f64 is exact below 2^53), or a magnitude i64 cannot hold.
    fn to_i64_exact(&self, truncate: bool) -> Option<i64> {
        if self.is_relaxed() {
            return None;
        }
        let exp = (self.sign_exponent & 0x7FFF) as i32;
        if exp == 0x7FFF {
            return None; // NaN / Inf -> indefinite, handled by the f64 path
        }
        let unbiased = exp - 0x3FFF;
        if !(0..=63).contains(&unbiased) {
            return None;
        }
        let mant = self.mantissa;
        if mant >> 63 == 0 {
            return None; // denormal / unnormal image: no implicit one to lean on
        }
        let sign = self.sign_exponent >> 15 != 0;
        let shift = (63 - unbiased) as u32;
        let int_part = mant >> shift;
        let frac = if shift == 0 { 0 } else { mant & ((1u64 << shift) - 1) };
        let mut mag = int_part;
        if frac != 0 && !truncate {
            let half = 1u64 << (shift - 1);
            let round_up = match rounding_mode() {
                1 => sign,  // toward -inf: away from zero only when negative
                2 => !sign, // toward +inf
                3 => false, // toward zero
                _ => frac > half || (frac == half && int_part & 1 == 1),
            };
            if round_up {
                mag = mag.checked_add(1)?;
            }
        }
        if sign {
            if mag > 1u64 << 63 {
                return None;
            }
            Some((mag as i64).wrapping_neg())
        } else {
            if mag > i64::MAX as u64 {
                return None;
            }
            Some(mag as i64)
        }
    }

    pub fn to_i32(&self) -> i32 {
        let f = f64::from_bits(self.to_f64());
        if f.is_nan() {
            return i32::MIN; // x87 indefinite integer
        }
        let rounded = round_by_rc(f);
        if rounded >= (i32::MAX as f64 + 1.0) || rounded < (i32::MIN as f64) {
            i32::MIN // overflow -> indefinite
        } else {
            rounded as i32
        }
    }

    pub fn to_i64(&self) -> i64 {
        if let Some(v) = self.to_i64_exact(false) {
            return v;
        }
        let f = f64::from_bits(self.to_f64());
        if f.is_nan() {
            return i64::MIN; // x87 indefinite integer
        }
        let rounded = round_by_rc(f);
        if rounded >= (i64::MAX as f64 + 1.0) || rounded < (i64::MIN as f64) {
            i64::MIN // overflow -> indefinite
        } else {
            rounded as i64
        }
    }

    pub fn truncate_to_i32(&self) -> i32 {
        let f = f64::from_bits(self.to_f64());
        if f.is_nan() || f >= (i32::MAX as f64 + 1.0) || f < (i32::MIN as f64) {
            i32::MIN
        } else {
            f as i32
        }
    }

    pub fn truncate_to_i64(&self) -> i64 {
        if let Some(v) = self.to_i64_exact(true) {
            return v;
        }
        let f = f64::from_bits(self.to_f64());
        if f.is_nan() || f >= (i64::MAX as f64 + 1.0) || f < (i64::MIN as f64) {
            i64::MIN
        } else {
            f as i64
        }
    }

    pub fn cos(self) -> F80 { F80::of_f64x(self.to_f64x().cos()) }
    pub fn sin(self) -> F80 { F80::of_f64x(self.to_f64x().sin()) }
    pub fn tan(self) -> F80 { F80::of_f64x(self.to_f64x().tan()) }
    pub fn atan(self) -> F80 { F80::of_f64x(self.to_f64x().atan()) }
    pub fn atan2(self, other: F80) -> F80 { F80::of_f64x(self.to_f64x().atan2(other.to_f64x())) }

    pub fn log2(self) -> F80 { F80::of_f64x(self.to_f64x().log2()) }
    pub fn ln(self) -> F80 { F80::of_f64x(self.to_f64x().ln()) }

    pub fn abs(self) -> F80 {
        if self.is_relaxed() {
            // Relaxed format: sign is bit 63 of mantissa (f64 bits)
            F80 { mantissa: self.mantissa & !(1u64 << 63), sign_exponent: RELAXED_TAG }
        } else {
            F80 { mantissa: self.mantissa, sign_exponent: self.sign_exponent & !0x8000 }
        }
    }
    pub fn two_pow(self) -> F80 { F80::of_f64x(2.0f64.powf(self.to_f64x())) }

    // Resolve the relaxed f64-bits form to a real 80-bit value; true F80 passes through.
    pub fn to_true_f80(self) -> F80 {
        if self.is_relaxed() {
            F80::of_f64_strict(self.mantissa)
        } else {
            self
        }
    }

    // FSCALE core: self * 2^n by adding to the exponent field. Doing this as
    // self * 2.0f64.powf(n) overflows to f64 inf for n > 1023, but the real f80
    // range goes to 2^16383.
    pub fn scale_pow2(self, n: i64) -> F80 {
        let v = self.to_true_f80();
        let sign_bit = v.sign_exponent & 0x8000;
        let exp_field = (v.sign_exponent & 0x7FFF) as i64;

        // Zero stays zero; Inf/NaN unchanged
        if (exp_field == 0 && v.mantissa == 0) || exp_field == 0x7FFF {
            return v;
        }

        let new_exp = exp_field + n;
        // Overflow -> signed Inf (cap below RELAXED_TAG so the result can't alias it)
        if new_exp >= 0x7FFE {
            return F80 { mantissa: 0x8000000000000000, sign_exponent: sign_bit | 0x7FFF };
        }
        // Underflow -> signed zero
        if new_exp <= 0 {
            return F80 { mantissa: 0, sign_exponent: sign_bit };
        }
        F80 { mantissa: v.mantissa, sign_exponent: sign_bit | (new_exp as u16) }
    }

    pub fn round(self) -> F80 {
        let f = self.to_f64x();
        let rounded = round_by_rc(f);
        F80::of_f64x(rounded)
    }

    pub fn trunc(self) -> F80 {
        F80::of_f64x(self.to_f64x().trunc())
    }

    pub fn sqrt(self) -> F80 {
        F80::of_f64x(apply_precision(self.to_f64x().sqrt()))
    }

    pub fn is_finite(self) -> bool {
        self != F80::POS_INFINITY && self != F80::NEG_INFINITY
    }
    pub fn is_nan(self) -> bool {
        self != self
    }

    pub fn get_exception_flags() -> u8 { 0 }
    pub fn clear_exception_flags() {}

    pub fn partial_cmp_quiet(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let a = f64::from_bits(self.to_f64());
        let b = f64::from_bits(other.to_f64());
        a.partial_cmp(&b)
    }
}

fn round_ties_even(f: f64) -> f64 {
    let rounded = f.round();
    // Check for tie case: fractional part is exactly 0.5
    let diff = (f - rounded).abs();
    if diff == 0.0 {
        // Not a tie, or already rounded correctly
        return rounded;
    }
    // f.round() rounds ties away from zero in Rust.
    // For ties-to-even, check if we need to adjust.
    let frac = f.abs() % 1.0;
    if (frac - 0.5).abs() < 1e-15 {
        // It's a tie - round to even
        let candidate = f.round();
        if (candidate as i64) % 2 != 0 {
            // Rounded to odd, go the other way
            if f > 0.0 { candidate - 1.0 } else { candidate + 1.0 }
        } else {
            candidate
        }
    } else {
        rounded
    }
}

impl std::ops::Add for F80 {
    type Output = F80;
    fn add(self, other: Self) -> Self {
        // Fast path: both operands already hold raw f64 bits
        if self.both_relaxed(&other) {
            let r = apply_precision(f64::from_bits(self.mantissa) + f64::from_bits(other.mantissa));
            return F80 { mantissa: r.to_bits(), sign_exponent: RELAXED_TAG };
        }
        let a = f64::from_bits(self.to_f64());
        let b = f64::from_bits(other.to_f64());
        F80::of_f64(apply_precision(a + b).to_bits())
    }
}
impl std::ops::Sub for F80 {
    type Output = F80;
    fn sub(self, other: Self) -> Self {
        if self.both_relaxed(&other) {
            let r = apply_precision(f64::from_bits(self.mantissa) - f64::from_bits(other.mantissa));
            return F80 { mantissa: r.to_bits(), sign_exponent: RELAXED_TAG };
        }
        let a = f64::from_bits(self.to_f64());
        let b = f64::from_bits(other.to_f64());
        F80::of_f64(apply_precision(a - b).to_bits())
    }
}
impl std::ops::Neg for F80 {
    type Output = F80;
    fn neg(self) -> Self {
        if self.is_relaxed() {
            // Relaxed format: sign is bit 63 of mantissa (f64 bits)
            F80 { mantissa: self.mantissa ^ (1u64 << 63), sign_exponent: RELAXED_TAG }
        } else {
            let mut result = self;
            result.sign_exponent ^= 1 << 15;
            result
        }
    }
}
impl std::ops::Mul for F80 {
    type Output = F80;
    fn mul(self, other: Self) -> Self {
        if self.both_relaxed(&other) {
            let r = apply_precision(f64::from_bits(self.mantissa) * f64::from_bits(other.mantissa));
            return F80 { mantissa: r.to_bits(), sign_exponent: RELAXED_TAG };
        }
        let a = f64::from_bits(self.to_f64());
        let b = f64::from_bits(other.to_f64());
        F80::of_f64(apply_precision(a * b).to_bits())
    }
}
impl std::ops::Div for F80 {
    type Output = F80;
    fn div(self, other: Self) -> Self {
        if self.both_relaxed(&other) {
            let r = apply_precision(f64::from_bits(self.mantissa) / f64::from_bits(other.mantissa));
            return F80 { mantissa: r.to_bits(), sign_exponent: RELAXED_TAG };
        }
        let a = f64::from_bits(self.to_f64());
        let b = f64::from_bits(other.to_f64());
        F80::of_f64(apply_precision(a / b).to_bits())
    }
}
impl std::ops::Rem for F80 {
    type Output = F80;
    fn rem(self, other: Self) -> Self {
        if self.both_relaxed(&other) {
            let r = f64::from_bits(self.mantissa) % f64::from_bits(other.mantissa);
            return F80 { mantissa: r.to_bits(), sign_exponent: RELAXED_TAG };
        }
        let quot = (self / other).trunc();
        self - quot * other
    }
}

impl PartialEq for F80 {
    fn eq(&self, other: &Self) -> bool {
        if self.both_relaxed(other) {
            return f64::from_bits(self.mantissa) == f64::from_bits(other.mantissa);
        }
        let a = f64::from_bits(self.to_f64());
        let b = f64::from_bits(other.to_f64());
        a == b
    }
}
impl PartialOrd for F80 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.both_relaxed(other) {
            return f64::from_bits(self.mantissa).partial_cmp(&f64::from_bits(other.mantissa));
        }
        let a = f64::from_bits(self.to_f64());
        let b = f64::from_bits(other.to_f64());
        a.partial_cmp(&b)
    }
}
