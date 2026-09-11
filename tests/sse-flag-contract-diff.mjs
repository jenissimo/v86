import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import {runSseFlagContract} from './sse-flag-contract-core.mjs';
const mulpdMode=process.env.V86_INLINE_MULPD||'1';
await runSseFlagContract({
    wasm:process.env.V86_WASM_PATH||fileURLToPath(new URL('../build/v86.wasm',import.meta.url)),
    inlineMove:process.env.V86_INLINE_SSE_MOVE!=='0',
    inlineMulpd:mulpdMode==='1',
    guardedMulpd:mulpdMode==='guarded',
    reference:process.env.V86_SSE_REFERENCE?fs.readFileSync(process.env.V86_SSE_REFERENCE,'utf8').split(/\r?\n/).filter(l=>l.startsWith('{')).map(JSON.parse):null,
});
