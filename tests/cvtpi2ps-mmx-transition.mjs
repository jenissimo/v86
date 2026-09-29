// CVTPI2PS enters MMX state only for its MMX-register form (like CVTPI2PD).
//
// The m64 form reads memory, not an MMX register, and must leave the x87 tag word and TOS
// alone. Transitioning there marks the whole x87 stack valid, so the next FLD overflows and
// loads the QNaN indefinite. SSE code that converts from memory and then returns to x87 math
// without an EMMS (NFSU2's audio resampler) turned every mix into NaN = silence.
//
//   node tests/cvtpi2ps-mmx-transition.mjs
import {fileURLToPath} from 'node:url';
import {V86} from '../build/libv86.mjs';
import {SHIPPING_JIT} from '../../../tools/jit-config/shipping.mjs';
const wasm=process.env.V86_WASM_PATH||fileURLToPath(new URL('../build/v86.wasm',import.meta.url));
const N=100000,BASE=0x100000,INTS=BASE+0x800,OUT=0x180000; // OUT off the code page: a store there would read as SMC and keep the page cold
const ONE=0x3f800000;
function image(){
    const b=new Uint8Array(4096),d=new DataView(b.buffer);
    [0x1badb002,0x10000,(-0x1badb002-0x10000)>>>0,BASE,BASE,BASE+4096,BASE+4096,BASE+0x40].forEach((v,i)=>d.setUint32(i*4,v,true));
    d.setInt32(INTS-BASE,3,true);d.setInt32(INTS-BASE+4,-5,true);
    let p=0x40;const e=(...x)=>{b.set(x,p);p+=x.length;},u=x=>{d.setUint32(p,x>>>0,true);p+=4;};
    e(0xbc);u(0x200000);                                   // mov esp
    e(0x0f,0x20,0xc0);e(0x25);u(~4);e(0x0d);u(2);e(0x0f,0x22,0xc0); // CR0: EM off, MP on
    e(0x0f,0x20,0xe0);e(0x0d);u(0x600);e(0x0f,0x22,0xe0);  // CR4: OSFXSR|OSXMMEXCPT
    e(0x31,0xff);e(0x31,0xf6);                             // EDI = error bits, ESI = iterations
    const loop=p;
    e(0xdb,0xe3);                                          // fninit
    e(0x0f,0x2a,0x05);u(INTS);                             // cvtpi2ps xmm0, [INTS]
    e(0xf3,0x0f,0x2c,0xc0);e(0x83,0xf8,3);e(0x74,0x03);e(0x83,0xcf,1); // cvttss2si eax,xmm0; != 3 -> bit0
    e(0xd9,0xe8);e(0xd9,0x1d);u(OUT);                      // fld1; fstp dword [OUT]
    e(0x81,0x3d);u(OUT);u(ONE);e(0x74,0x03);e(0x83,0xcf,2); // x87 disturbed by m64 form -> bit1
    e(0xdb,0xe3);                                          // fninit
    e(0x0f,0x2a,0xc8);                                     // cvtpi2ps xmm1, mm0 (register form)
    e(0xd9,0xe8);e(0xd9,0x1d);u(OUT);                      // fld1 -> must overflow
    e(0xa1);u(OUT);e(0x25);u(0x7f800000);e(0x3d);u(0x7f800000); // exponent all ones = the overflow NaN
    e(0x74,0x03);e(0x83,0xcf,4);                           // no transition -> bit2
    e(0x46);e(0x81,0xfe);u(N);e(0x0f,0x82);u(loop-(p+4));
    e(0xdb,0xe3);e(0x89,0xf8);e(0xf4);return b;
}
async function run(jit){
    const em=new V86({autostart:false,memory_size:16<<20,wasm_path:wasm,log_level:0,...(jit?{}:{disable_jit:1})});
    await new Promise(r=>em.add_listener('emulator-loaded',r));
    const c=em.v86.cpu,w=c.wm.exports;c.reboot_internal();c.reset_memory();c.load_multiboot(image().buffer);
    if(jit)for(const [i,v]of SHIPPING_JIT)w.set_jit_config(i,v);
    globalThis.__wasmDump={out:[]};
    await new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>{em.stop();reject(Error('timeout'));},30000);
        em.bus.register('cpu-event-halt',()=>{clearTimeout(timer);em.stop();resolve();});em.run();
    });
    const errors=c.reg32[0]>>>0,iterations=c.reg32[6]>>>0;
    if(iterations!==N)throw Error(`jit=${jit}: ran ${iterations}/${N} iterations`);
    if(errors&1)throw Error(`jit=${jit}: cvtpi2ps m64 converted wrong`);
    if(errors&2)throw Error(`jit=${jit}: cvtpi2ps m64 disturbed the x87 stack (MMX transition on the memory form)`);
    if(errors&4)throw Error(`jit=${jit}: cvtpi2ps mm lost its MMX transition`);
    if(jit&&!globalThis.__wasmDump.out.some(r=>WebAssembly.Module.imports(new WebAssembly.Module(r.bytes)).some(x=>x.name==='instr_0F2A_reg')))
        throw Error('jit=1: no compiled block calls instr_0F2A_reg — the JIT path was not exercised');
    em.destroy();
    console.log(`PASS jit=${jit}`);
}
await run(0);
await run(1);
