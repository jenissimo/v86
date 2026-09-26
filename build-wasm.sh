#!/usr/bin/env bash
# Build v86 WASM and libv86.mjs from source
# Requirements: Rust (wasm32-unknown-unknown target), clang, Node.js, Java
set -euo pipefail

# Prefer rustup over Homebrew Rust (Homebrew doesn't have wasm32 target)
export PATH="$HOME/.cargo/bin:$PATH"

cd "$(dirname "$0")"

# Auto-detect LLVM on Windows (winget installs to "C:/Program Files/LLVM/bin")
if ! command -v clang &>/dev/null; then
    for d in "/c/Program Files/LLVM/bin" "C:/Program Files/LLVM/bin"; do
        if [ -x "$d/clang" ] || [ -x "$d/clang.exe" ]; then
            export PATH="$d:$PATH"
            break
        fi
    done
fi

echo "=== Generating Rust instruction tables ==="
node gen/generate_jit.js --output-dir build/ --table jit
node gen/generate_jit.js --output-dir build/ --table jit0f
node gen/generate_interpreter.js --output-dir build/ --table interpreter
node gen/generate_interpreter.js --output-dir build/ --table interpreter0f
node gen/generate_analyzer.js --output-dir build/ --table analyzer
node gen/generate_analyzer.js --output-dir build/ --table analyzer0f
node gen/generate_flag_contracts.js

echo "=== Compiling C dependencies (zstd) ==="
mkdir -p build

clang -c -Wall \
    --target=wasm32 -O3 -flto -nostdlib -fvisibility=hidden -ffunction-sections -fdata-sections \
    -DZSTDLIB_VISIBILITY="" \
    -o build/zstddeclib.o \
    lib/zstd/zstddeclib.c

echo "=== Building v86.wasm (Rust → WASM) ==="
# Use .cmd wrapper on Windows, original script on Unix
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    LINKER_WRAPPER=tools/rust-lld-wrapper.cmd
else
    LINKER_WRAPPER=tools/rust-lld-wrapper
fi

cargo rustc --release \
    --target wasm32-unknown-unknown \
    -- \
    -C linker=$LINKER_WRAPPER \
    -C "link-args=--import-table --global-base=4096" \
    -C "link-args=build/zstddeclib.o" \
    -C target-feature=+bulk-memory \
    -C target-feature=+multivalue \
    -C target-feature=+simd128

cp build/wasm32-unknown-unknown/release/v86.wasm build/v86.wasm
echo "  → build/v86.wasm ($(du -h build/v86.wasm | cut -f1))"

# Tripwire: #[no_mangle] exports contributed by workspace crates (crates/d3d9-webgpu)
# must survive LTO/linking into the final cdylib. Fails the build if any went missing.
node tools/check-wasm-exports.mjs build/v86.wasm

# BottleShip: Vite serves v86.wasm from public/ (dev) and dist/ (prod build).
# Keeping them in sync with vendor/v86/build/v86.wasm is easy to forget and
# manifests as "Missing import: <name>" at worker startup when the built
# libv86.mjs references a Rust export that the stale .wasm doesn't have.
# Mirror automatically. Both locations are project-relative; the build-wasm
# script is invoked from the repo root (or the script's own dir — `cd
# "$(dirname "$0")"` was done above), so compute paths relative to that.
REPO_ROOT="$(cd ../.. && pwd)"
for dest in "$REPO_ROOT/public/v86.wasm" "$REPO_ROOT/dist/v86.wasm"; do
    if [ -f "$dest" ] || [ -d "$(dirname "$dest")" ]; then
        cp build/v86.wasm "$dest"
        echo "  → $dest"
    fi
done

echo "=== Building libv86.mjs (Closure Compiler) ==="
CLOSURE=closure-compiler/compiler.jar
if [ ! -f "$CLOSURE" ]; then
    echo "  Downloading Closure Compiler..."
    mkdir -p closure-compiler
    curl -sL -o "$CLOSURE" https://repo1.maven.org/maven2/com/google/javascript/closure-compiler/v20210601/closure-compiler-v20210601.jar
fi

CORE_FILES="src/cjs.js src/const.js src/io.js src/main.js src/lib.js src/buffer.js src/ide.js src/pci.js src/floppy.js src/dma.js src/pit.js src/vga.js src/ps2.js src/rtc.js src/uart.js src/acpi.js src/iso9660.js src/state.js src/ne2k.js src/sb16.js src/virtio.js src/virtio_console.js src/virtio_net.js src/virtio_balloon.js src/bus.js src/log.js src/cpu.js src/elf.js src/kernel.js"
BROWSER_FILES="src/browser/screen.js src/browser/keyboard.js src/browser/mouse.js src/browser/speaker.js src/browser/serial.js src/browser/network.js src/browser/starter.js src/browser/worker_bus.js src/browser/dummy_screen.js src/browser/inbrowser_network.js src/browser/fake_network.js src/browser/wisp_network.js src/browser/fetch_network.js src/browser/print_stats.js src/browser/filestorage.js"
LIB_FILES="lib/9p.js lib/filesystem.js lib/marshall.js"

JS_ARGS=""
for f in $CORE_FILES $BROWSER_FILES $LIB_FILES; do
    JS_ARGS="$JS_ARGS --js $f"
done

java -jar "$CLOSURE" \
    --js_output_file build/libv86.mjs \
    --define=DEBUG=false \
    --generate_exports \
    --externs src/externs.js \
    --warning_level VERBOSE \
    --jscomp_error accessControls \
    --jscomp_error checkRegExp \
    --jscomp_error checkTypes \
    --jscomp_error checkVars \
    --jscomp_error conformanceViolations \
    --jscomp_error const \
    --jscomp_error constantProperty \
    --jscomp_error deprecated \
    --jscomp_error deprecatedAnnotations \
    --jscomp_error duplicateMessage \
    --jscomp_error es5Strict \
    --jscomp_error externsValidation \
    --jscomp_error globalThis \
    --jscomp_error invalidCasts \
    --jscomp_error misplacedTypeAnnotation \
    --jscomp_error missingProperties \
    --jscomp_error missingReturn \
    --jscomp_error msgDescriptions \
    --jscomp_error nonStandardJsDocs \
    --jscomp_error suspiciousCode \
    --jscomp_error strictModuleDepCheck \
    --jscomp_error typeInvalidation \
    --jscomp_error undefinedVars \
    --jscomp_error unknownDefines \
    --jscomp_error visibility \
    --use_types_for_optimization \
    --assume_function_wrapper \
    --summary_detail_level 3 \
    --language_in ECMASCRIPT_2020 \
    --language_out ECMASCRIPT_2020 \
    --compilation_level SIMPLE \
    --jscomp_off=missingProperties \
    --output_wrapper ";let module = {exports:{}}; %output%; export default module.exports.V86; export let {V86, CPU} = module.exports;" \
    $JS_ARGS \
    --chunk_output_type=ES_MODULES \
    --emit_use_strict=false

echo "  → build/libv86.mjs ($(du -h build/libv86.mjs | cut -f1))"

echo "=== Done! ==="
