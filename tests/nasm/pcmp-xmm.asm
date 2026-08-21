; PCMPEQB/W/D (660F74/75/76) and PCMPGTB/W/D (660F64/65/66) on xmm.
;
; The comparisons are SIGNED, and their result is an all-ones or all-zeroes mask
; per element. The operands include pairs that order one way unsigned and the
; other way signed (0x80 vs 0x7f, 0xff vs 0x01), so a translation that reaches
; for an unsigned WASM comparison inverts those lanes and nothing else.
global _start

section .data
	align 16
; bytes: 00 01 7f 80 ff fe 10 20  aa aa 00 ff 7f 80 01 02
a:	dq	0x2010feff807f0100
	dq	0x0201807fff00aaaa
; bytes: 00 02 80 7f 01 ff 10 21  aa 55 ff 00 80 7f 02 01
b:	dq	0x2110ff017f800200
	dq	0x01027f8000ff55aa

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

	pcmpeqb		xmm1, xmm6
	pcmpeqw		xmm2, [b]
	pcmpgtb		xmm3, xmm6
	pcmpgtw		xmm4, [b]
	pcmpgtd		xmm5, xmm6
	; aliasing: equal to itself everywhere, greater than itself nowhere
	pcmpgtb		xmm7, xmm7

%include "footer.inc"
