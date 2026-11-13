# C ABI 共享库：一份实现，Go 和 Node.js 都加载

> 回答两个问题：
> 1. **能不能编译 DLL 然后让 Go / Node 用？** 能，但前提是**接口必须是 C ABI**；
> 2. **对 `.so` 适用吗？性能有影响吗？** 完全适用（同一套代码，Linux 编 `.so`、Windows 编 `.dll`）；
>    性能影响可量化，见下方实测。

## 结论先行

| 问题 | 答案 |
|---|---|
| Go 能加载 `.so` / `.dll` 吗？ | **能**。Linux/macOS：cgo + `dlopen/dlsym`；Windows：`syscall.NewLazyDLL`（纯 Go，无需 cgo）或 cgo `LoadLibrary` |
| Node 能加载 `.so` / `.dll` 吗？ | 不能直接 `require` 裸库，**必须包一层 N-API addon（`.node`）** 或使用 FFI 库（koffi/ffi-napi） |
| 接口能用 Go interface / JS 对象吗？ | **不行**。跨语言只能用 C 类型（`int` / `char*` / buffer / 函数指针） |
| `.so` 和 `.dll` 是两套代码吗？ | 不是。同一份 `plugin.c`，只换编译目标 |
| 性能影响？ | 每次跨语言调用约 **30~45 ns**；批量计算摊薄后 ~**0.2 ns/元素**；`dlopen` 一次性 ~**2~8 ms** |

## 平台映射

| 宿主 | Linux | Windows | macOS |
|---|---|---|---|
| 共享库 | `.so` | `.dll` | `.dylib` |
| Go 加载 | `cgo` + `dlopen/dlsym`（本 demo 的 `gohost/`） | `syscall.NewLazyDLL` + `NewProc`（`gohost-win/`） | 同 Linux |
| Node 加载 | N-API addon `dlopen`（`nodehost/addon.c`） | N-API addon `LoadLibraryA`（同一份代码 `#ifdef _WIN32`） | 同 Linux |
| 产物 ABI | C ABI（稳定） | C ABI（稳定） | C ABI |

> 重要区分：
> - **Go 标准库 `plugin` 不是这个**：`plugin.Open` 只能加载「Go 编译出来的 `.so`」，不能加载 C ABI 库，而且**不支持 Windows**。
> - **Node 的 `require()` 不是直接加载 DLL**：CJS/ESM 只加载 JS 模块，`.node` 才是原生扩展入口（`process.dlopen`）。

## 实测结果（本机 Linux，数值仅作相对比较）

```bash
bash abi-plugin/run.sh
```

```text
== 4. 同一个 .so，Go 宿主调用
[go] dlopen 耗时: 1.98ms（一次性）
[go] name=greet-abi interface_version=1 add(1,2)=3
[go] run("world") => hello, world (C ABI plugin)
[go] 基准:
[go]   pure Go add      : 0.22 ns/op
[go]   cgo→C add        : 44.0 ns/op  （每次跨语言边界）
[go]   1 次调用算 1e6 次: 0.22 ns/element（批量摊薄后）

== 5. 同一个 .so，Node 宿主调用
[node] addon + dlopen 耗时: 7.82 ms（一次性）
[node] name=greet-abi interface_version=1 add(1,2)=3
[node] run("world") => hello, world (C ABI plugin)
[node] 基准:
[node]   pure JS add       : 1.51 ns/op
[node]   N-API → C add     : 29.5 ns/op  （每次跨语言边界）
[node]   1 次调用算 1e6 次 : 0.23 ns/element（批量摊薄后）
```

### 性能怎么解读

| 场景 | 开销 | 是否在意 |
|---|---|---|
| 每次调用做有意义的工作（> 1 µs，如 IO、编解码、算法） | 跨边界 30~45 ns | **可以忽略** |
| 高频紧密循环里调用极小函数（百万次/秒） | 每次 30~45 ns，比原生调用慢 1~2 个数量级 | **需要合并成批量接口**（一次调用传入数组/缓冲区，C 里循环） |
| 加载/卸载 | `dlopen` 2~8 ms（一次性）；重复加载可忽略 | 常驻进程只加载一次即可 |
| 字符串/JSON 跨边界 | 序列化/拷贝成本通常 > 边界成本 | 大 payload 用二进制 buffer 或共享内存 |
| 崩溃隔离 | **没有**：C 段崩溃 = 宿主崩溃 | 需要隔离就用 RPC 子进程 |

经验规则：**边界调用次数少、单次负载大 → C ABI 共享库很划算；调用次数极多且单次极轻 → 走批量 API 或留在同语言**。

## 怎么编译

```bash
# Linux .so（本机可运行）
gcc -shared -fPIC -O2 -o build/libgreet.so plugin.c

# Windows .dll（交叉编译，本机只验证导出表）
x86_64-w64-mingw32-gcc -shared -O2 -o build/greet.dll plugin.c

# Node N-API addon（Linux）
gcc -shared -fPIC -O2 -I"$(dirname "$(dirname "$(node -p 'process.execPath')")")/include/node" \
    -o build/greeter.node nodehost/addon.c -ldl

# Go 宿主（cgo）
(cd gohost && go build -o ../build/gohost .)

# Windows Go 消费者（无 cgo）
(cd gohost-win && GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o ../build/gohost-win.exe .)

# 反向：Go 也能编 DLL 给别人用
(cd go-export && GOOS=windows GOARCH=amd64 CGO_ENABLED=1 CC=x86_64-w64-mingw32-gcc \
   go build -buildmode=c-shared -o ../build/greet-go.dll .)
```

Windows 实机上还需要：

- **Node addon**：`nodehost/binding.gyp` 已经准备好，在 Windows 上执行
  ```bat
  cd abi-plugin\nodehost
  npx node-gyp rebuild
  ```
  得到 `build/Release/greeter.node`（`addon.c` 里的 `#ifdef _WIN32` 会自动改用 `LoadLibraryA`），
  然后 `set ABI_PLUGIN_PATH=..\build\greet.dll` 再运行 Node 脚本。N-API 保证跨 Node 版本 ABI 稳定，但仍需按 `win32-x64` 编译；
- **Go 消费者**：`syscall.NewLazyDLL("greet.dll")`，不需要 cgo、不需要头文件，把 `gohost-win.exe` 和 `greet.dll` 放同一目录直接运行；
- 也可以不做原生 addon，直接用 FFI 库（如 `koffi`、`ffi-napi`）在 JS 里声明 `greet.dll` 的函数签名，少写 C，但多一个依赖。

## 目录

```
abi-plugin/
├── plugin.c          # C ABI 实现（一份代码，Linux/Windows 通用）
├── gohost/           # Go 消费者：cgo dlopen（Linux）
├── gohost-win/       # Go 消费者：syscall.NewLazyDLL（Windows，交叉编译验证）
├── nodehost/
│   ├── addon.c       # Node N-API 包装（dlopen / LoadLibraryA）
│   └── bench.mjs     # Node 侧调用 + 基准
├── go-export/        # 反向：Go 用 -buildmode=c-shared 编 DLL
├── run.sh            # 一键构建 + 运行 + 交叉编译验证
└── build/            # 构建产物（gitignore）
```

## 和「Go plugin / Node 动态 import」方案的关系

- 如果你的模块就在同一门语言里（Go 模块给 Go 宿主、JS 模块给 Node 宿主）：不用 C ABI，直接用 `plugin` / `import()` 更简单、更快（同语言调用 ~ns 级）。
- 如果希望**一份二进制被多语言共用**（比如 C 核心算法同时给 Go 服务 + Node 脚本 + Python 用）：C ABI 共享库是最省事的路子。
- 如果还要**崩溃隔离 / 热替换 / 跨机器**：再上一层到 RPC 子进程（gRPC / JSON-RPC），C ABI 只解决「同进程跨语言调用」。