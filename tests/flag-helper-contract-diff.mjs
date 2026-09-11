// Dirty arithmetic flags must survive FPU-only helpers. FCOMI must still replace them.
import {fileURLToPath} from 'node:url';
import {V86} from '../build/libv86.mjs';
import {SHIPPING_JIT} from '../../../tools/jit-config/shipping.mjs';
const wasm=process.env.V86_WASM_PATH||fileURLToPath(new URL('../build/v86.wasm',import.meta.url));
const N=100000,BASE=0x100000;
function image(op, writesFlags){
    const b=new Uint8Array(4096),d=new DataView(b.buffer);
    [0x1badb002,0x10000,(-0x1badb002-0x10000)>>>0,BASE,BASE,BASE+4096,BASE+4096,BASE+0x40].forEach((v,i)=>d.setUint32(i*4,v,true));
    let p=0x40;const e=(...x)=>{b.set(x,p);p+=x.length;},u=x=>{d.setUint32(p,x>>>0,true);p+=4;};
    e(0xbc);u(0x200000);e(0x31,0xff);e(0x31,0xf6); // ESP, error accumulator EDI, iterations ESI
    const loop=p;
    e(0xdb,0xe3);e(0xd9,0xe8);e(0xd9,0xe8); // FNINIT; FLD1; FLD1
    e(0xb8);u(0x7fffffff);e(0x83,0xc0,1); // ADD: OF/SF/AF/PF set, CF/ZF clear
    e(...op);
    e(0x9c);e(0x58); // PUSHFD; POP EAX (observe lazy flags at a helper boundary)
    e(0x25);u(0x8d5);e(0x35);u(writesFlags?0x40:0x894);e(0x09,0xc7); // accumulate any incorrect flag
    e(0x46);e(0x81,0xfe);u(N);e(0x0f,0x82);u(loop-(p+4));
    e(0x89,0xf8);e(0xf4);return b;
}
async function run(op,writesFlags,relaxed,locals){
    const em=new V86({autostart:false,memory_size:16<<20,wasm_path:wasm,log_level:0});
    await new Promise(r=>em.add_listener('emulator-loaded',r));
    const c=em.v86.cpu,w=c.wm.exports;c.reboot_internal();c.reset_memory();c.load_multiboot(image(op,writesFlags).buffer);
    for(const [i,v]of SHIPPING_JIT)w.set_jit_config(i,v);
    w.set_relaxed_fpu(relaxed);w.set_jit_config(21,locals);
    globalThis.__wasmDump={out:[]};
    await new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>{em.stop();reject(Error('timeout'));},20000);
        em.bus.register('cpu-event-halt',()=>{clearTimeout(timer);em.stop();resolve();});em.run();
    });
    const modules=globalThis.__wasmDump.out;
    if(!modules.length||c.reg32[0]!==0||c.reg32[6]!==N)throw Error(`flags/work mismatch: ${JSON.stringify({op,relaxed,locals,error:c.reg32[0],iterations:c.reg32[6],modules:modules.length})}`);
    if(!writesFlags && !modules.some(r=>WebAssembly.Module.imports(new WebAssembly.Module(r.bytes)).some(x=>x.name===`instr16_D9_${op[1]<0xf8?6:7}_reg`)))throw Error('tested helper absent from emitted code');
    if(writesFlags && !relaxed && !modules.some(r=>WebAssembly.Module.imports(new WebAssembly.Module(r.bytes)).some(x=>x.name==='fpu_fcomi')))throw Error('FCOMI writer helper absent from emitted code');
    em.destroy();
}
for(const relaxed of [0,1]){
    for(let opcode=0xf0;opcode<=0xff;opcode++)for(const locals of [0,1])await run([0xd9,opcode],false,relaxed,locals);
    for(const locals of [0,1])await run([0xdb,0xf1],true,relaxed,locals);
    console.log(`PASS relaxed=${relaxed}: 16 preserving operations + FCOMI writer, idx21 OFF/ON`);
}
console.log('PASS 68 helper/flags cases');
