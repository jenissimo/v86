// Hot MOVHPS/MOVHPD fault and restart; inspect state at #PF before replay.
import {V86} from '../build/libv86.mjs';
import {SHIPPING_JIT} from '../../../tools/jit-config/shipping.mjs';
export async function runSseFaultContract({wasm,mulpd=false,inline=true,log=console.log}){
const BASE=0x100000,PD=0x108000,PT=0x109000,FAULT=0x200000,VALID=0x210000,DATA=0x220000,N=60000;
function image(prefix,offset){
    const b=new Uint8Array(4096),d=new DataView(b.buffer);
    [0x1badb002,0x10000,(-0x1badb002-0x10000)>>>0,BASE,BASE,BASE+4096,BASE+4096,BASE+0x40].forEach((v,i)=>d.setUint32(i*4,v,true));
    b.set([0,0,0,0,0,0,0,0,255,255,0,0,0,0x9a,0xcf,0,255,255,0,0,0,0x92,0xcf,0],0xc00);
    d.setUint16(0xc20,23,true);d.setUint32(0xc22,BASE+0xc00,true);
    const gate=0xc30+14*8,H=BASE+0x800;
    d.setUint16(gate,H&65535,true);d.setUint16(gate+2,8,true);b[gate+5]=0x8e;d.setUint16(gate+6,H>>>16,true);
    d.setUint16(0xcb8,127,true);d.setUint32(0xcba,BASE+0xc30,true);
    let p=0x40;const e=(...x)=>{b.set(x,p);p+=x.length;},u=x=>{d.setUint32(p,x>>>0,true);p+=4;};
    e(0x0f,0x01,0x15);u(BASE+0xc20);e(0xea);u(BASE+p+6);e(8,0);
    e(0x66,0xb8,0x10,0,0x8e,0xd8,0x8e,0xc0,0x8e,0xd0);e(0xbc);u(0x300000);
    e(0x0f,0x01,0x1d);u(BASE+0xcb8);
    e(0xb8);u(PD);e(0x0f,0x22,0xd8);e(0x0f,0x20,0xc0);e(0x0d);u(0x80000000);e(0x0f,0x22,0xc0);
    e(0x0f,0x20,0xe0);e(0x0d);u(0x600);e(0x0f,0x22,0xe0);
    e(0x0f,0x10,0x05);u(DATA); // original XMM low and high
    e(0xc7,0x05);u(PT+(FAULT>>>12)*4);u(0);e(0x0f,0x01,0x3d);u(FAULT);
    e(0xb9);u(N);
    const loop=p;e(0xba);u(VALID);e(0x83,0xf9,1,0x75,5);e(0xba);u(FAULT-offset);
    e(0xbb);u(0xc0deface);e(0xb8);u(0x7fffffff);e(0x83,0xc0,1);
    const faultIP=BASE+p;if(prefix)e(0x66);e(0x0f,mulpd?0x59:0x16,0x02);
    e(0x9c,0x5f,0x49,0x0f,0x85);u(loop-(p+4));
    e(0x0f,0x11,0x05);u(DATA+80);e(0xf4);
    p=0x800;e(0x50);e(0xa3);u(DATA+32);e(0x89,0x1d);u(DATA+36);
    e(0x0f,0x20,0xd0);e(0xa3);u(DATA+40);
    for(const [off,dest]of [[16,44],[8,48],[4,52]]){e(0x8b,0x44,0x24,off);e(0xa3);u(DATA+dest);}
    e(0x0f,0x11,0x05);u(DATA+64);e(0xff,0x05);u(DATA+56);
    e(0xc7,0x05);u(PT+(FAULT>>>12)*4);u(FAULT|3);e(0x0f,0x01,0x3d);u(FAULT);
    e(0x58,0x83,0xc4,4,0xcf);
    return {b,faultIP};
}
for(const prefix of mulpd?[true]:[false,true])for(const offset of Array.from({length:mulpd?16:8},(_,i)=>i))for(const mode of ['interpreter','jit']){
    const im=image(prefix,offset),em=new V86({autostart:false,memory_size:16<<20,wasm_path:wasm,log_level:0});
    await new Promise(r=>em.add_listener('emulator-loaded',r));
    const c=em.v86.cpu,w=c.wm.exports;c.reboot_internal();c.reset_memory();c.load_multiboot(im.b.buffer);
    for(let i=0;i<1024;i++){c.write32(PD+i*4,0);c.write32(PT+i*4,(i<<12)|3);}c.write32(PD,PT|3);
    [0x01234567,0x89abcdef,0x10101010,0x20202020].forEach((v,i)=>c.write32(DATA+i*4,v));
    c.write32(VALID,0x33445566);c.write32(VALID+4,0x778899aa);
    c.write32(FAULT-offset,0xaabbccdd);c.write32(FAULT-offset+4,0xeeff0123);
    if(mulpd){
        [0,0x40000000,0,0x40080000].forEach((v,i)=>c.write32(DATA+i*4,v)); // 2,3
        [0,0x3ff00000,0,0x3ff00000].forEach((v,i)=>c.write32(VALID+i*4,v)); // 1,1
        [0,0x40100000,0,0x40140000].forEach((v,i)=>c.write32(FAULT-offset+i*4,v)); // 4,5
    }
    for(const [i,v]of SHIPPING_JIT)w.set_jit_config(i,v);w.set_jit_config(0,mode==='interpreter'?1:0);
    let jitFaults=0,jitFaultIP=0;
    const deliver=c.jit_imports.trigger_fault_end_jit;
    c.jit_imports.trigger_fault_end_jit=()=>{jitFaults++;jitFaultIP=c.instruction_pointer[0]>>>0;return deliver();};
    globalThis.__wasmDump={out:[]};
    await new Promise((resolve,reject)=>{const t=setTimeout(()=>{em.stop();reject(Error('timeout'));},20000);em.bus.register('cpu-event-halt',()=>{clearTimeout(t);em.stop();resolve();});em.run();});
    const read=a=>c.read32s(a)>>>0,words=a=>Array.from({length:4},(_,i)=>read(a+i*4));
    const got={eax:read(DATA+32),ebx:read(DATA+36),cr2:read(DATA+40),flags:read(DATA+44)&0x8d5,eip:read(DATA+48),error:read(DATA+52),faults:read(DATA+56),before:words(DATA+64),after:words(DATA+80),remaining:c.reg32[1],finalFlags:c.reg32[7]&0x8d5};
    const want={eax:0x80000000,ebx:0xc0deface,cr2:FAULT,flags:0x894,eip:im.faultIP,error:0,faults:1,before:[0x01234567,0x89abcdef,0x33445566,0x778899aa],after:[0x01234567,0x89abcdef,0xaabbccdd,0xeeff0123],remaining:0,finalFlags:0x894};
    if(mulpd){want.before=[0,0x40000000,0,0x40080000];want.after=[0,0x40200000,0,0x402e0000];}
    if(JSON.stringify(got)!==JSON.stringify(want))throw Error(JSON.stringify({mulpd,prefix,offset,mode,got,want}));
    if(mode==='jit'&&(jitFaults!==1||jitFaultIP!==im.faultIP))throw Error(`fault not delivered by JIT at target: ${jitFaults}/${jitFaultIP}`);
    if(mode==='interpreter'&&jitFaults!==0)throw Error('interpreter unexpectedly entered JIT fault helper');
    if(mode==='jit'){
        const mods=globalThis.__wasmDump.out;if(!mods.length)throw Error('JIT absent');
        const calls=mods.some(r=>WebAssembly.Module.imports(new WebAssembly.Module(r.bytes)).some(x=>x.name===(mulpd?'instr_660F59':'instr_0F16')));
        if(calls===inline)throw Error('unexpected SSE helper presence');
    }
    em.destroy();log(`PASS ${mulpd?'MULPD':prefix?'MOVHPD':'MOVHPS'} offset=${offset} ${mode}`);
}
log('PASS 32 page-fault/restart cases');
return {count:32};
}
