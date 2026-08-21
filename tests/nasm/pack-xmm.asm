; PACKUSWB (660F67) / PACKSSWB (660F63) / PACKSSDW (660F6B) on xmm.
;
; These are the only ops in the integer-SSE set whose operand order flips under
; a naive translation to a WASM narrowing instruction: narrow_i16x8_*(a, b) puts
; `a` in the low lanes, but x86 puts the DESTINATION in the low lanes. Getting it
; backwards swaps the two halves of every packed vector -- a wrong image that
; still looks like an image. So the operands here are two disjoint ascending
; ranges: the correct result is 01..08 then 11..18, and a swap reads 11..18
; first, which cannot be mistaken for anything else.
global _start

section .data
	align 16
; words 0001 0002 0003 0004 0005 0006 0007 0008 -> bytes 01..08
lo:	dq	0x0004000300020001
	dq	0x0008000700060005
; words 0011 0012 0013 0014 0015 0016 0017 0018 -> bytes 11..18
hi:	dq	0x0014001300120011
	dq	0x0018001700160015
; saturation fodder: words ffff(-1) 8000 7fff 0100 00ff 0080 fffe 0001
sat:	dq	0x01007fff8000ffff
	dq	0x0001fffe008000ff
; dwords 00000001 00000002 00000003 00000004 (for packssdw)
dlo:	dq	0x0000000200000001
	dq	0x0000000400000003
; dwords ffffffff 00008000 00007fff 7fffffff
dsat:	dq	0x00008000ffffffff
	dq	0x7fffffff00007fff

%include "header.inc"

	pxor		xmm1, xmm1
	pxor		xmm2, xmm2
	pxor		xmm3, xmm3
	pxor		xmm4, xmm4
	pxor		xmm5, xmm5
	pxor		xmm6, xmm6
	pxor		xmm7, xmm7

	movdqa		xmm1, [lo]
	movdqa		xmm2, [lo]
	movdqa		xmm3, [sat]
	movdqa		xmm4, [sat]
	movdqa		xmm5, [dlo]
	movdqa		xmm6, [hi]
	movdqa		xmm7, [sat]

	; ORDER: expect bytes 01..08 (from xmm1) then 11..18 (from xmm6)
	packuswb	xmm1, xmm6
	; same, mem form
	packsswb	xmm2, [hi]
	; SATURATION: unsigned pack clamps negatives to 0 and >255 to ff
	packuswb	xmm3, [sat]
	; signed pack clamps to 80..7f
	packsswb	xmm4, [sat]
	; dword -> word pack, order and saturation
	packssdw	xmm5, [dsat]
	; aliasing: both halves identical
	packuswb	xmm7, xmm7

%include "footer.inc"
