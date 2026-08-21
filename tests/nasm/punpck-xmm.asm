; PUNPCK{L,H}{BW,WD,DQ,QDQ} on xmm: lane ORDER, which is what a shuffle-mask
; translation gets wrong. The operands are two disjoint ascending byte ranges,
; so any interleave that swaps source/dest or reverses a pair is visible by
; inspection rather than as a plausible-looking number.
global _start

section .data
	align 16
; bytes 00 01 02 ... 0f  (this is the destination)
a:	dq	0x0706050403020100
	dq	0x0f0e0d0c0b0a0908
; bytes 10 11 12 ... 1f  (this is the source)
b:	dq	0x1716151413121110
	dq	0x1f1e1d1c1b1a1918

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

	; expect bytes 00 10 01 11 02 12 03 13 04 14 05 15 06 16 07 17
	punpcklbw	xmm1, xmm6
	; expect bytes 08 18 09 19 0a 1a 0b 1b 0c 1c 0d 1d 0e 1e 0f 1f
	punpckhbw	xmm2, xmm6
	; mem forms of the same two
	punpcklwd	xmm3, [b]
	punpckhwd	xmm4, [b]
	punpckldq	xmm5, [b]
	; aliasing: duplicates each byte of the low half
	punpcklbw	xmm7, xmm7

%include "footer.inc"
