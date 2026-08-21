BITS 32
    ; ANDNPS/PANDN: expect an inline v128.andnot whose FIRST operand is the
    ; source and whose second is the destination, because x86 negates the
    ; destination while v128.andnot negates its second argument.
    andnps xmm1, xmm2
    andnps xmm1, [esi]
    pandn  xmm3, xmm4
    hlt
