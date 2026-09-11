// Arithmetic flags span actual SSE helpers; check register, memory and alias forms.
import { V86 } from '../build/libv86.mjs';
import { SHIPPING_JIT } from '../../../tools/jit-config/shipping.mjs';
export async function runSseFlagContract({wasm,inlineMove=true,inlineMulpd=false,guardedMulpd=false,reference=null,log=console.log}){
const nanRecords=[];
const BASE=0x100000,DATA=BASE+0x800,OUT=DATA+32,N=100000;
// Current source inlines moves. Set V86_INLINE_SSE_MOVE=0 for an older helper-based artifact.
const cases=[
    {name:'movlhps',op:[0x0f,0x16,0xc1],expected:[2,4],helper:'instr_0F16'},
    {name:'movlhps-alias',op:[0x0f,0x16,0xc0],expected:[2,2],helper:'instr_0F16'},
    {name:'movhps-mem',op:[0x0f,0x16,0x05],mem:true,expected:[2,4],helper:'instr_0F16'},
    {name:'movhpd-mem',op:[0x66,0x0f,0x16,0x05],mem:true,expected:[2,4],helper:'instr_0F16'},
    {name:'movlhps-bits',op:[0x0f,0x16,0xc1],bits:[0,0x80000000,1,0x7ff00000,0x12345678,0x7ff81234,0,0],expectedBits:[0,0x80000000,0x12345678,0x7ff81234],helper:'instr_0F16'},
    {name:'movhps-mem-bits',op:[0x0f,0x16,0x05],mem:true,bits:[0,0x80000000,0,0,1,0x7ff00000,0,0],expectedBits:[0,0x80000000,1,0x7ff00000],helper:'instr_0F16'},
    {name:'mulpd',op:[0x66,0x0f,0x59,0xc1],expected:[8,15],helper:'instr_660F59'},
    {name:'mulpd-alias',op:[0x66,0x0f,0x59,0xc0],expected:[4,9],helper:'instr_660F59'},
    {name:'mulpd-mem',op:[0x66,0x0f,0x59,0x05],mem:true,expected:[8,15],helper:'instr_660F59'},
];
for(const mem of [false,true]){
    const op=mem?[0x66,0x0f,0x59,0x05]:[0x66,0x0f,0x59,0xc1];
    cases.push({name:`mulpd-signed-zero-infinity-${mem?'mem':'reg'}`,op,mem,
        bits:[0,0,0,0xfff00000,0,0xc0000000,0,0x3fe00000],expectedBits:[0,0x80000000,0,0xfff00000],helper:'instr_660F59'});
    cases.push({name:`mulpd-overflow-underflow-${mem?'mem':'reg'}`,op,mem,
        bits:[0xffffffff,0x7fefffff,1,0,0,0x40000000,0,0x3fe00000],expectedBits:[0,0x7ff00000,0,0],helper:'instr_660F59'});
    for(const [name,bits] of [
        ['nan-source',[0,0x40000000,0,0x40080000,0x12345678,0x7ff81234,1,0x7ff00000]],
        ['nan-destination',[0x87654321,0xfff85678,2,0xfff00000,0,0x40000000,0,0x40080000]],
        ['nan-both',[0x87654321,0xfff85678,2,0xfff00000,0x12345678,0x7ff81234,1,0x7ff00000]],
        ['invalid-zero-infinity',[0,0,0,0x80000000,0,0x7ff00000,0,0xfff00000]],
    ])cases.push({name:`mulpd-${name}-${mem?'mem':'reg'}`,op,mem,bits,nanReference:true,nanBoth:name==='nan-both',helper:'instr_660F59'});
}
cases.push({name:'mulpd-nan-alias',op:[0x66,0x0f,0x59,0xc0],bits:[0x12345678,0x7ff81234,1,0xfff00000,0,0,0,0],nanReference:true,helper:'instr_660F59'});
function image(test){
    const b=new Uint8Array(4096),d=new DataView(b.buffer);
    [0x1badb002,0x10000,(-0x1badb002-0x10000)>>>0,BASE,BASE,BASE+4096,BASE+4096,BASE+0x40].forEach((v,i)=>d.setUint32(i*4,v,true));
    [2,3,4,5].forEach((v,i)=>d.setFloat64(DATA-BASE+i*8,v,true));
    test.bits?.forEach((v,i)=>d.setUint32(DATA-BASE+i*4,v,true));
    let p=0x40;const e=(...x)=>{b.set(x,p);p+=x.length;},u=x=>{d.setUint32(p,x>>>0,true);p+=4;};
    e(0xbc);u(0x200000);e(0x31,0xff);e(0x31,0xf6);
    e(0x0f,0x20,0xe0);e(0x0d);u(0x600);e(0x0f,0x22,0xe0); // CR4.OSFXSR/OSXMMEXCPT
    const loop=p;
    e(0x66,0x0f,0x10,0x05);u(DATA);e(0x66,0x0f,0x10,0x0d);u(DATA+16);
    e(0xb8);u(0x7fffffff);e(0x83,0xc0,1);
    e(...test.op);if(test.mem)u(DATA+16);
    e(0x9c,0x58);e(0x25);u(0x8d5);e(0x35);u(0x894);e(0x09,0xc7);
    e(0x46);e(0x81,0xfe);u(N);e(0x0f,0x82);u(loop-(p+4));
    e(0x66,0x0f,0x11,0x05);u(OUT);e(0xf4);return b;
}
for(const test of cases)for(const mode of ['interpreter','off','on']){
    const em=new V86({autostart:false,memory_size:16<<20,wasm_path:wasm,log_level:0});
    await new Promise(r=>em.add_listener('emulator-loaded',r));
    const c=em.v86.cpu,w=c.wm.exports;c.reboot_internal();c.reset_memory();c.load_multiboot(image(test).buffer);
    for(const [i,v]of SHIPPING_JIT)w.set_jit_config(i,v);
    w.set_jit_config(0,mode==='interpreter'?1:0);w.set_jit_config(21,mode==='on'?1:0);
    globalThis.__wasmDump={out:[]};
    let mulpdCalls=0;
    const mulpdHelper=c.jit_imports.instr_660F59;
    c.jit_imports.instr_660F59=(...args)=>{mulpdCalls++;return mulpdHelper(...args);};
    await new Promise((resolve,reject)=>{const t=setTimeout(()=>{em.stop();reject(Error('timeout'));},20000);em.bus.register('cpu-event-halt',()=>{clearTimeout(t);em.stop();resolve();});em.run();});
    const d=new DataView(new ArrayBuffer(16));
    test.expected?.forEach((v,i)=>d.setFloat64(i*8,v,true));
    if(test.nanReference&&mode==='interpreter'){
        // Payload choice is implementation-dependent in Wasm. Preserve the current
        // helper's observed bits on this engine, including quieting/sign behavior.
        test.expectedBits=Array.from({length:4},(_,i)=>c.read32s(OUT+i*4)>>>0);
        for(const high of [test.expectedBits[1],test.expectedBits[3]])
            if((high&0x7ff80000)!==0x7ff80000)throw Error('reference did not produce quiet NaN');
    }
    const expected=test.expectedBits||Array.from({length:4},(_,i)=>d.getUint32(i*4,true));
    const actual=Array.from({length:4},(_,i)=>c.read32s(OUT+i*4)>>>0);
    if(test.nanReference){const record={case:test.name,mode,nanBits:actual};nanRecords.push(record);log(JSON.stringify(record));}
    if(reference&&test.nanReference){
        const ref=reference.find(x=>x.case===test.name&&x.mode===mode);
        if(!ref||JSON.stringify(actual)!==JSON.stringify(ref.nanBits))throw Error(`${test.name}/${mode} differs from baseline NaN bits`);
    }
    if(test.nanBoth){
        // Existing helper JIT and interpreter already differ in which input NaN
        // wins. Check quieting and input payload preservation without inventing
        // a deterministic preference absent from the baseline.
        for(const lane of [0,2]){
            const allowed=[lane,lane+4].map(i=>[test.bits[i],(test.bits[i+1]|0x80000)>>>0]);
            if(!allowed.some(([lo,hi])=>actual[lane]===lo&&actual[lane+1]===hi))
                throw Error(`${test.name}/${mode} unexpected NaN ${actual}`);
        }
    }else for(let i=0;i<4;i++)if(actual[i]!==expected[i])throw Error(`${test.name}/${mode} word ${i}`);
    if(c.reg32[7]!==0||c.reg32[6]!==N)throw Error(`${test.name}/${mode} flags=${c.reg32[7]} iterations=${c.reg32[6]}`);
    if(mode!=='interpreter'){
        const modules=globalThis.__wasmDump.out;
        if(!modules.length)throw Error('no JIT modules');
        const called=modules.some(r=>WebAssembly.Module.imports(new WebAssembly.Module(r.bytes)).some(x=>x.name===test.helper));
        const shouldInline=(inlineMove&&test.helper==='instr_0F16')||(inlineMulpd&&test.helper==='instr_660F59');
        if(called===shouldInline)throw Error('unexpected helper presence');
        if(guardedMulpd&&test.helper==='instr_660F59'){
            const nonFinite=[1,3,5,7].some(i=>((test.bits?.[i]||0)&0x7ff00000)===0x7ff00000);
            if(nonFinite?mulpdCalls===0:mulpdCalls!==0)throw Error(`${test.name}: unexpected fallback count ${mulpdCalls}`);
            log(`MULPD fallback ${test.name}/${mode}: ${mulpdCalls}`);
        }
    }
    em.destroy();log(`PASS ${test.name}/${mode}`);
}
log(`PASS ${cases.length*3} SSE/flags cases`);
return {count:cases.length*3,nanRecords};
}
