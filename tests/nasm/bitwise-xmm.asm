; ANDPS/ORPS/XORPS/ANDNPS (0F54..0F57) on xmm, reg and mem forms.
;
; ANDNPS is the asymmetric one: the result is (NOT dest) AND src, so a
; translation to v128.andnot -- whose own argument order is andnot(a, b) = a AND
; NOT b -- has to swap the operands. The operands below are chosen so every
; combination of (a AND NOT b) and (NOT a AND b) is a distinct value.
global _start

section .data
	align 16
a:	dq	0x00000000ffffffff
	dq	0xf0f0f0f00f0f0f0f
b:	dq	0x0000ffff0000ffff
	dq	0xff00ff00ff00ff00

%include "header.inc"

	pxor		xmm1, xmm1
	pxor		xmm2, xmm2
	pxor		xmm3, xmm3
	pxor		xmm4, xmm4
	pxor		xmm5, xmm5
	pxor		xmm6, xmm6
	pxor		xmm7, xmm7

	movdqa		xmm1, [a]
	movdqa		xmm2, [a]
	movdqa		xmm3, [a]
	movdqa		xmm4, [a]
	movdqa		xmm5, [a]
	movdqa		xmm6, [b]
	movdqa		xmm7, [a]

	; reg forms
	andps		xmm1, xmm6
	orps		xmm2, xmm6
	xorps		xmm3, xmm6
	; ANDNPS: (NOT xmm4) AND xmm6, NOT the other way round
	andnps		xmm4, xmm6
	; mem form of the asymmetric one
	andnps		xmm5, [b]
	; aliasing: (NOT x) AND x is zero for every bit
	andnps		xmm7, xmm7

%include "footer.inc"
