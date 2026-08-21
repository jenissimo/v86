BITS 32
    ; PADDSW: expect an inline i16x8.add_sat_s, reg and mem forms,
    ; instead of a call to $e.instr_660FED.
    paddsw xmm1, xmm2
    paddsw xmm1, [esi]
    hlt
