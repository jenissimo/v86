import {fileURLToPath} from 'node:url';
import {runSseFaultContract} from './sse-move-pagefault-core.mjs';
const mulpd=process.argv.includes('--mulpd');
await runSseFaultContract({
    wasm:process.env.V86_WASM_PATH||fileURLToPath(new URL('../build/v86.wasm',import.meta.url)),
    mulpd,
    inline:mulpd?(process.env.V86_INLINE_MULPD||'1')==='1':process.env.V86_INLINE_SSE_MOVE!=='0',
});
