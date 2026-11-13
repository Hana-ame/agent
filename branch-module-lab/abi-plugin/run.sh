#!/usr/bin/env bash
# 演示：同一份 C ABI 共享库，被 Go 和 Node.js 分别加载调用。
#
#   Linux  : 编译成 .so  → Go 用 cgo dlopen；Node 用 N-API addon 包一层
#   Windows: 编译成 .dll → Go 用 LoadLibrary/GetProcAddress（或 syscall.NewLazyDLL）；
#                           Node 用 N-API addon（内部 LoadLibraryA）
#
# 本环境是 Linux，所以：
#   - .so 路径真实运行（Go + Node 都跑通）
#   - .dll 用 MinGW 交叉编译出来，只检查导出表（Windows DLL 无法在 Linux 执行）
set -euo pipefail
cd "$(dirname "$0")"

NODE_INC="$(dirname "$(dirname "$(node -p 'process.execPath')")")/include/node"
mkdir -p build

echo "== 1. 编译 C ABI 共享库（Linux .so）"
gcc -shared -fPIC -O2 -o build/libgreet.so plugin.c
file build/libgreet.so | sed 's/^/   /'

echo
echo "== 2. 编译 Node N-API 包装（.node，内部 dlopen 上面的 .so）"
gcc -shared -fPIC -O2 -I"$NODE_INC" -o build/greeter.node nodehost/addon.c -ldl
file build/greeter.node | sed 's/^/   /'

echo
echo "== 3. 编译 Go 宿主（cgo dlopen，无链接期依赖）"
(cd gohost && go build -o ../build/gohost .)
file build/gohost | sed 's/^/   /'

echo
echo "== 4. 同一个 .so，Go 宿主调用"
./build/gohost -lib "$PWD/build/libgreet.so"

echo
echo "== 5. 同一个 .so，Node 宿主调用"
ABI_PLUGIN_PATH="$PWD/build/libgreet.so" node nodehost/bench.mjs

echo
echo "== 6. 交叉编译 Windows DLL（MinGW），确认导出符号"
x86_64-w64-mingw32-gcc -shared -O2 -o build/greet.dll plugin.c
file build/greet.dll | sed 's/^/   /'
x86_64-w64-mingw32-objdump -p build/greet.dll | sed -n '/Ordinal\/Name Pointer/,/^$/p' | sed 's/^/   /'

echo
echo "== 7. Windows 侧消费者：交叉编译验证（不执行）"
(cd gohost-win && GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o ../build/gohost-win.exe .)
file build/gohost-win.exe | sed 's/^/   /'
(cd go-export && GOOS=windows GOARCH=amd64 CGO_ENABLED=1 CC=x86_64-w64-mingw32-gcc \
  go build -buildmode=c-shared -o ../build/greet-go.dll .)
file build/greet-go.dll | sed 's/^/   /'
x86_64-w64-mingw32-objdump -p build/greet-go.dll | sed -n '/Ordinal\/Name Pointer/,/^$/p' | sed 's/^/   /'

echo
echo "结论：一份 C ABI 实现，Go / Node 都能加载；Linux 用 .so，Windows 换 .dll 即可，调用方式对应 dlopen/LoadLibrary。"