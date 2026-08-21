; PSRLQ / PSLLQ (quadword) and PSRLDQ / PSLLDQ (whole-register BYTE shifts) by
; imm8 on xmm.
;
; PSRLDQ/PSLLDQ shift by BYTES, not bits, and a count > 15 clears the register
; outright -- v86 gets that for free today because the count indexes past a
; zero-initialized result, which is exactly the kind of correct-by-construction
; behaviour a translation to an explicit shuffle loses. The quadword shifts have
; the same count-vs-width hazard as the word shifts.
global _start

section .data
	align 16
; qwords: 8000000000000001, ffffffff0000ffff
v:	dq	0x8000000000000001
	dq	0xffffffff0000ffff
; bytes 00 01 02 ... 0f, so a byte shift is readable as a displacement
seq:	dq	0x0706050403020100
	dq	0x0f0e0d0c0b0a0908

%include "header.inc"

	pxor		xmm1, xmm1
	pxor		xmm2, xmm2
	pxor		xmm3, xmm3
	pxor		xmm4, xmm4
	pxor		xmm5, xmm5
	pxor		xmm6, xmm6
	pxor		xmm7, xmm7

	; PSRLQ / PSLLQ: ordinary counts, then the width boundary and beyond
	movdqa		xmm1, [v]
	psrlq		xmm1, 1
	movdqa		xmm2, [v]
	psrlq		xmm2, 63
	movdqa		xmm3, [v]
	psrlq		xmm3, 64
	movdqa		xmm4, [v]
	psllq		xmm4, 64
	movdqa		xmm5, [v]
	psllq		xmm5, 100

	; PSRLDQ / PSLLDQ: byte displacements, then a count past the register
	movdqa		xmm6, [seq]
	psrldq		xmm6, 3
	movdqa		xmm7, [seq]
	pslldq		xmm7, 5

	movdqa		xmm0, [seq]
	psrldq		xmm0, 16

%include "footer.inc"
