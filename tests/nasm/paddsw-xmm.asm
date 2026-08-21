; PADDSW (660FED) on xmm: signed-saturating word add at the exact clamp boundaries.
; The generated tests feed random words, which hit saturation only statistically;
; these pin both clamp directions, no-clamp, and the dest==src aliasing case.
global _start

section .data
	align 16
; words: 7fff 7fff 8000 8000 0002 0001 ffff ffff
a:	dq	0x800080007fff7fff
	dq	0xffffffff00010002
; words: 0001 0001 ffff ffff 0004 0003 0001 0000
b:	dq	0xffffffff00010001
	dq	0x0000000100030004
; words: 8000 8000 8000 8000 7fff 7fff 7fff 7fff
c:	dq	0x8000800080008000
	dq	0x7fff7fff7fff7fff

%include "header.inc"

	pxor		xmm1, xmm1
	pxor		xmm2, xmm2
	pxor		xmm3, xmm3
	pxor		xmm4, xmm4
	pxor		xmm5, xmm5
	pxor		xmm6, xmm6
	pxor		xmm7, xmm7

	movdqa		xmm1, [a]
	movdqa		xmm2, [b]
	movdqa		xmm3, [c]
	movdqa		xmm4, [a]
	movdqa		xmm5, [c]
	movdqa		xmm6, [c]

	; reg form
	paddsw		xmm1, xmm2
	; mem form
	paddsw		xmm4, [b]
	; both clamp directions in one vector
	paddsw		xmm3, [c]
	; aliasing: dest == src (the inline form loads both before storing)
	paddsw		xmm5, xmm5
	; adding zero must be the identity, including for 0x8000
	paddsw		xmm6, xmm0

%include "footer.inc"
