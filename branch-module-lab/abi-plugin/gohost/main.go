// Go 宿主：用 cgo + dlopen/dlsym 在运行期加载 C ABI 共享库（Linux .so / macOS .dylib）。
//
// 为什么不用标准库 plugin？plugin 只能加载「Go 编译出来的 .so」，不能加载 C ABI 库，
// 也不支持 Windows。要消费 .dll / .so，Go 侧就是 cgo（Linux）或 syscall.NewLazyDLL（Windows）。
package main

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdlib.h>

typedef int (*version_fn)(void);
typedef const char* (*name_fn)(void);
typedef int (*run_fn)(const char*, char*, int);
typedef int (*add_fn)(int, int);
typedef long long (*bench_fn)(int);

static version_fn g_version;
static name_fn g_name;
static run_fn g_run;
static add_fn g_add;
static bench_fn g_bench;

static int load_plugin(const char* path) {
	void* lib = dlopen(path, RTLD_NOW);
	if (!lib) return -1;
	g_version = (version_fn)dlsym(lib, "plugin_interface_version");
	g_name    = (name_fn)dlsym(lib, "plugin_name");
	g_run     = (run_fn)dlsym(lib, "plugin_run");
	g_add     = (add_fn)dlsym(lib, "add");
	g_bench   = (bench_fn)dlsym(lib, "bench_sum");
	return (g_version && g_name && g_run && g_add && g_bench) ? 0 : -2;
}
static int call_version(void) { return g_version(); }
static const char* call_name(void) { return g_name(); }
static int call_run(const char* in, char* out, int n) { return g_run(in, out, n); }
static int call_add(int a, int b) { return g_add(a, b); }
static long long call_bench(int n) { return g_bench(n); }
*/
import "C"

import (
	"flag"
	"fmt"
	"time"
	"unsafe"
)

func add(a, b int) int { return a + b }

func main() {
	lib := flag.String("lib", "./build/libgreet.so", "C ABI 共享库路径")
	flag.Parse()

	path := C.CString(*lib)
	defer C.free(unsafe.Pointer(path))

	t0 := time.Now()
	rc := C.load_plugin(path)
	loadTime := time.Since(t0)
	if rc != 0 {
		panic(fmt.Sprintf("加载 %s 失败（rc=%d）", *lib, rc))
	}
	fmt.Printf("[go] dlopen 耗时: %v（一次性）\n", loadTime)
	fmt.Printf("[go] name=%s interface_version=%d add(1,2)=%d\n",
		C.GoString(C.call_name()), int(C.call_version()), int(C.call_add(1, 2)))

	in := C.CString("world")
	out := (*C.char)(C.malloc(512))
	defer C.free(unsafe.Pointer(in))
	defer C.free(unsafe.Pointer(out))
	if rc := C.call_run(in, out, 512); rc != 0 {
		panic("plugin_run failed")
	}
	fmt.Printf("[go] run(\"world\") => %s\n", C.GoString(out))

	// 性能：每次跨 cgo 边界 vs 一次调用里批量算
	const N = 1_000_000
	var s, s2 int
	t1 := time.Now()
	for i := 0; i < N; i++ {
		s += add(1, 2)
	}
	pureGo := time.Since(t1)

	t2 := time.Now()
	for i := 0; i < N; i++ {
		s2 += int(C.call_add(1, 2))
	}
	cgo := time.Since(t2)

	t3 := time.Now()
	_ = C.call_bench(N)
	batch := time.Since(t3)

	fmt.Printf("[go] 基准（本机，仅相对比较）:\n")
	fmt.Printf("[go]   pure Go add      : %.2f ns/op\n", float64(pureGo.Nanoseconds())/N)
	fmt.Printf("[go]   cgo→C add        : %.1f ns/op  （每次跨语言边界）\n", float64(cgo.Nanoseconds())/N)
	fmt.Printf("[go]   1 次调用算 1e6 次: %.2f ns/element（批量摊薄后）\n", float64(batch.Nanoseconds())/N)
	_, _ = s, s2
}