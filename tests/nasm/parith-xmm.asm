; The remaining integer-SSE arithmetic worth pinning before it is translated:
; PADDB/W/D, PSUBB/W/D (wrapping), PADDSB/PSUBSB (signed byte saturate),
; PMAXUB/PMINUB (unsigned byte), PAVGB (rounded average).
;
; PAVGB is the subtle one: the average rounds UP, i.e. (a + b + 1) >> 1 computed
; at a width that cannot overflow. Dropping the +1 or truncating to 8 bits mid-way
; is off by one on exactly half the inputs, which reads as a slightly dark image
; rather than as a failure.
global _start

section .data
	align 16
; bytes: 00 01 7f 80 ff fe 10 20  ff 01 80 7f 55 aa 03 fd
a:	dq	0x2010feff807f0100
	dq	0xfd03aa557f8001ff
; bytes: 01 ff 01 ff 01 02 f0 e0  01 ff 7f 80 aa 55 fd 03
b:	dq	0xe0f00201ff01ff01
	dq	0x03fd55aa807fff01

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

	; wrapping adds/subs, reg and mem
	paddb		xmm1, xmm6
	psubb		xmm1, xmm6
	paddw		xmm1, [b]
	psubd		xmm2, xmm6
	; signed byte saturation, both directions
	paddsb		xmm3, xmm6
	psubsb		xmm4, [b]
	; unsigned byte min/max
	movdqa		xmm5, [a]
	pmaxub		xmm5, xmm6
	pminub		xmm5, [b]
	; rounded unsigned byte average
	pavgb		xmm7, xmm6
	pavgw		xmm7, [b]

%include "footer.inc"
