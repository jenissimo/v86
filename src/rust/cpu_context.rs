use crate::cpu::memory;
use crate::prefix::{PREFIX_MASK_ADDRSIZE, PREFIX_MASK_OPSIZE};
use crate::state_flags::CachedStateFlags;

// Decode bound for compile-time instruction analysis. While set (page-tail mode only), a read
// that would reach past it is not performed: it returns 0 and latches DECODE_CROSSED, so an
// instruction straddling a page end is recognised WITHOUT fetching a byte of the next physical
// page — which need not be the next virtual page, may be unmapped, or may be MMIO.
static mut DECODE_LIMIT: u32 = u32::MAX;
static mut DECODE_CROSSED: bool = false;

/// Run `f` with decoding bounded to `[.., limit)`; returns false when `f` tried to read past it.
pub fn decode_within<T>(limit: u32, f: impl FnOnce() -> T) -> (T, bool) {
    unsafe {
        DECODE_LIMIT = limit;
        DECODE_CROSSED = false;
    }
    let r = f();
    let ok = unsafe { !DECODE_CROSSED };
    unsafe {
        DECODE_LIMIT = u32::MAX;
        DECODE_CROSSED = false;
    }
    (r, ok)
}

#[derive(Clone)]
pub struct CpuContext {
    pub eip: u32,
    pub prefixes: u8,
    pub cs_offset: u32,
    pub state_flags: CachedStateFlags,
}

impl CpuContext {
    pub fn advance16(&mut self) {
        dbg_assert!(self.eip & 0xFFF <= 0x1000 - 2);
        self.eip += 2;
    }
    pub fn advance32(&mut self) {
        dbg_assert!(self.eip & 0xFFF <= 0x1000 - 4);
        self.eip += 4;
    }
    #[allow(unused)]
    pub fn advance_moffs(&mut self) {
        if self.asize_32() {
            self.advance32()
        }
        else {
            self.advance16()
        }
    }

    #[inline]
    fn past_decode_limit(&mut self, n: u32) -> bool {
        unsafe {
            if self.eip as u64 + n as u64 > DECODE_LIMIT as u64 {
                DECODE_CROSSED = true;
                self.eip += n;
                return true;
            }
        }
        false
    }

    pub fn read_imm8(&mut self) -> u8 {
        if self.past_decode_limit(1) {
            return 0;
        }
        let v = memory::read8(self.eip) as u8;
        self.eip += 1;
        v
    }
    pub fn read_imm8s(&mut self) -> i8 { self.read_imm8() as i8 }
    pub fn read_imm16(&mut self) -> u16 {
        if self.past_decode_limit(2) {
            return 0;
        }
        dbg_assert!(self.eip & 0xFFF <= 0x1000 - 2);
        let v = memory::read16(self.eip) as u16;
        self.eip += 2;
        v
    }
    pub fn read_imm32(&mut self) -> u32 {
        if self.past_decode_limit(4) {
            return 0;
        }
        dbg_assert!(self.eip & 0xFFF <= 0x1000 - 4);
        let v = memory::read32s(self.eip) as u32;
        self.eip += 4;
        v
    }
    pub fn read_moffs(&mut self) -> u32 {
        if self.asize_32() {
            self.read_imm32()
        }
        else {
            self.read_imm16() as u32
        }
    }

    pub fn cpl3(&self) -> bool { self.state_flags.cpl3() }
    pub fn has_flat_segmentation(&self) -> bool { self.state_flags.has_flat_segmentation() }
    pub fn osize_32(&self) -> bool {
        self.state_flags.is_32() != (self.prefixes & PREFIX_MASK_OPSIZE != 0)
    }
    pub fn asize_32(&self) -> bool {
        self.state_flags.is_32() != (self.prefixes & PREFIX_MASK_ADDRSIZE != 0)
    }
    pub fn ssize_32(&self) -> bool { self.state_flags.ssize_32() }
}
