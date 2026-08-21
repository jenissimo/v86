; PMULLW (660FD5) on xmm: low 16 bits of the word product, high half discarded.
; Pins the cases where a wrong width or a signed/unsigned mix-up would show:
; 0x8000*0x8000, 0xffff*0xffff, and products that overflow into the high half.
global _start

section .data
	align 16
; words: 8000 ffff 0002 7fff  0100 0101 000f fffe
a:	dq	0x7fff0002ffff8000
	dq	0xfffe000f01010100
; words: 8000 ffff 4000 0002  0100 0101 1111 0003
b:	dq	0x00024000ffff8000
	dq	0x0003111101010100

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
	movdqa		xmm3, [a]
	movdqa		xmm4, [a]

	; reg form
	pmullw		xmm1, xmm2
	; mem form
	pmullw		xmm3, [b]
	; aliasing: squares, incl. 0x8000*0x8000 == 0 and 0xffff*0xffff == 1
	pmullw		xmm4, xmm4
	; multiplying by zero must clear every lane
	pmullw		xmm2, xmm0

%include "footer.inc"
