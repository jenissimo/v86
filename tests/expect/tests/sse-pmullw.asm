BITS 32
    ; PMULLW: expect an inline i16x8.mul, reg and mem forms.
    pmullw xmm3, xmm4
    pmullw xmm3, [esi]
    hlt
