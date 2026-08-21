BITS 32
    ; PSUBUSB: expect an inline i8x16.sub_sat_u with the DESTINATION as the
    ; first operand -- the op is not commutative.
    psubusb xmm7, xmm0
    psubusb xmm7, [esi]
    hlt
