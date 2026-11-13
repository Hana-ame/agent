/*
 * 一份 C ABI 实现，编译成 Linux .so 或 Windows .dll 后，Go / Node.js 都能加载。
 *
 * ABI 约定（跨语言唯一要守的东西）：
 *   - 只用 C 类型：int / char* / 定长 buffer；不要传 Go interface、JS 对象；
 *   - 字符串用「调用方给 buffer + 长度」的方式，避免跨语言 new/free 不对称；
 *   - 导出符号名稳定，加 interface_version 便于演进；
 *   - 不要跨边界抛异常、不要回调进宿主（复杂场景走 RPC 更稳）。
 */
#include <stdint.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int plugin_interface_version(void) { return 1; }

EXPORT const char* plugin_name(void) { return "greet-abi"; }

/* 把 "hello, <input> (C ABI plugin)" 写进调用方提供的 buffer。
 * 返回 0 成功；-1 表示 buffer 太小。 */
EXPORT int plugin_run(const char* input, char* out, int out_len) {
    const char* parts[3] = {"hello, ", input, " (C ABI plugin)"};
    int n = 0;
    for (int p = 0; p < 3; p++) {
        for (const char* s = parts[p]; *s; s++) {
            if (n >= out_len - 1) return -1;
            out[n++] = *s;
        }
    }
    out[n] = '\0';
    return 0;
}

EXPORT int add(int a, int b) { return a + b; }

/* 用来测「一次跨语言调用 vs 多次调用」的开销差异 */
EXPORT long long bench_sum(int n) {
    long long s = 0;
    for (int i = 0; i < n; i++) s += i;
    return s;
}