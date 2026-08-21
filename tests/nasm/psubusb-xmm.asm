; PSUBUSB (660FD8) and PADDUSB (660FDC) on xmm.
; These two must NOT share a saturation helper: unsigned byte subtract can only
; underflow (clamp to 0) while unsigned byte add can only overflow (clamp to
; 0xff). A helper that clamps one side only is correct for one and silently
; wrong for the other, so both are pinned here against the same operands.
global _start

section .data
	align 16
; bytes: 00 01 7f 80 fe ff 10 20  30 40 50 60 70 80 90 a0
a:	dq	0x2010fffe807f0100
	dq	0xa090807060504030
; bytes: 01 00 80 7f ff fe 20 10  ff 01 00 ff 80 80 7f 60
b:	dq	0x1020feff7f800001
	dq	0x607f8080ff0001ff

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
	movdqa		xmm5, [b]
	movdqa		xmm6, [a]

	; PSUBUSB: clamps low only
	psubusb		xmm1, xmm5
	psubusb		xmm2, [b]
	; x - x == 0 for every lane
	psubusb		xmm6, xmm6
	; PADDUSB: clamps high only
	paddusb		xmm3, xmm5
	paddusb		xmm4, [b]

%include "footer.inc"
