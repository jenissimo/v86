#[no_mangle]
pub fn call_indirect1(f: fn(u16), x: u16) { f(x); }

#[no_mangle]
pub fn call_indirect1_i32(f: fn(i32), x: i32) { f(x); }
