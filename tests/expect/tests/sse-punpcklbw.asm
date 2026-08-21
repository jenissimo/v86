BITS 32
    ; PUNPCKLBW: expect an inline i8x16.shuffle whose lane immediates read
    ; 0 16 1 17 2 18 3 19 4 20 5 21 6 22 7 23 -- destination byte first.
    punpcklbw xmm5, xmm6
    punpcklbw xmm5, [esi]
    hlt
