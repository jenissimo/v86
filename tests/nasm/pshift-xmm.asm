; PSRAW / PSRLW / PSLLW by imm8 on xmm.
;
; The shift COUNT is where x86 and WASM disagree. x86 treats a count >= the
; element width as "shift everything out": the logical shifts produce zero and
; the arithmetic shift produces the sign fill. WASM's i16x8.shr_s and friends
; instead MASK the count modulo the element width, so a count of 16 becomes a
; no-op and a count of 17 becomes a shift by 1. A naive translation is therefore
; wrong for exactly the counts the generated random tests almost always pick.
;
; Conversely the generated tests draw imm8 uniformly from 0..255 and so almost
; never exercise a SMALL count, which is the ordinary case. Both regimes are
; pinned here: 0, 1, 15, 16, 17 and a large value, per shift.
global _start

section .data
	align 16
; words: 8001 7fff 0001 ffff  8000 0100 0002 f00f
v:	dq	0xffff00017fff8001
	dq	0xf00f000201008000

%include "header.inc"

	pxor		xmm1, xmm1
	pxor		xmm2, xmm2
	pxor		xmm3, xmm3
	pxor		xmm4, xmm4
	pxor		xmm5, xmm5
	pxor		xmm6, xmm6
	pxor		xmm7, xmm7

	; PSRAW: count 0 (identity), 1, 15 (sign fill from bit 15), then the
	; boundary and beyond, both of which must saturate to the sign fill.
	movdqa		xmm1, [v]
	psraw		xmm1, 0
	psraw		xmm1, 1
	movdqa		xmm2, [v]
	psraw		xmm2, 15
	movdqa		xmm3, [v]
	psraw		xmm3, 16
	movdqa		xmm4, [v]
	psraw		xmm4, 17
	movdqa		xmm5, [v]
	psraw		xmm5, 200

	; PSRLW / PSLLW: >= 16 must yield zero, not a masked re-shift.
	movdqa		xmm6, [v]
	psrlw		xmm6, 1
	psrlw		xmm6, 15
	movdqa		xmm7, [v]
	psrlw		xmm7, 16

	movdqa		xmm0, [v]
	psllw		xmm0, 16

%include "footer.inc"
