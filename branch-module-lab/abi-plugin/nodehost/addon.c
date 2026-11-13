/*
 * Node.js 侧加载 C ABI 共享库的最小 N-API 包装。
 *
 * Node 不能直接 require 一个 .dll/.so —— 必须先包一层 N-API addon（.node）。
 * addon 在 Init 时 LoadLibrary/dlopen 目标库并取符号，之后把函数暴露成 JS 函数。
 * 同一份 addon.c 在 Linux 编 .node 用 dlopen，在 Windows 编 .node 用 LoadLibraryA。
 */
#include <node_api.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE lib_handle;
static lib_handle open_lib(const char* path) { return LoadLibraryA(path); }
static void* get_sym(lib_handle h, const char* name) { return (void*)GetProcAddress(h, name); }
#define DEFAULT_LIB "greet.dll"
#else
#include <dlfcn.h>
typedef void* lib_handle;
static lib_handle open_lib(const char* path) { return dlopen(path, RTLD_NOW); }
static void* get_sym(lib_handle h, const char* name) { return dlsym(h, name); }
#define DEFAULT_LIB "./build/libgreet.so"
#endif

typedef int (*version_fn)(void);
typedef const char* (*name_fn)(void);
typedef int (*run_fn)(const char*, char*, int);
typedef int (*add_fn)(int, int);
typedef long long (*bench_fn)(int);

static version_fn g_version = NULL;
static name_fn g_name = NULL;
static run_fn g_run = NULL;
static add_fn g_add = NULL;
static bench_fn g_bench = NULL;

static napi_value Name(napi_env env, napi_callback_info info) {
  napi_value s;
  napi_create_string_utf8(env, g_name(), NAPI_AUTO_LENGTH, &s);
  return s;
}

static napi_value Version(napi_env env, napi_callback_info info) {
  napi_value v;
  napi_create_int32(env, g_version(), &v);
  return v;
}

static napi_value Add(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value argv[2];
  int32_t a = 0, b = 0;
  napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  napi_get_value_int32(env, argv[0], &a);
  napi_get_value_int32(env, argv[1], &b);
  napi_value r;
  napi_create_int32(env, g_add(a, b), &r);
  return r;
}

static napi_value BenchSum(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  int32_t n = 0;
  napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  napi_get_value_int32(env, argv[0], &n);
  napi_value r;
  napi_create_int32(env, (int32_t)g_bench(n), &r);
  return r;
}

static napi_value Run(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  char input[256];
  char out[512];
  size_t len = 0;
  napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  napi_get_value_string_utf8(env, argv[0], input, sizeof(input), &len);
  input[len] = '\0';
  if (g_run(input, out, sizeof(out)) != 0) {
    napi_throw_error(env, NULL, "plugin_run failed (buffer too small?)");
    return NULL;
  }
  napi_value s;
  napi_create_string_utf8(env, out, NAPI_AUTO_LENGTH, &s);
  return s;
}

NAPI_MODULE_INIT() {
  const char* path = getenv("ABI_PLUGIN_PATH");
  if (!path) path = DEFAULT_LIB;
  lib_handle lib = open_lib(path);
  if (!lib) {
    napi_throw_error(env, NULL, "cannot load ABI plugin (check ABI_PLUGIN_PATH)");
    return exports;
  }
  g_version = (version_fn)get_sym(lib, "plugin_interface_version");
  g_name = (name_fn)get_sym(lib, "plugin_name");
  g_run = (run_fn)get_sym(lib, "plugin_run");
  g_add = (add_fn)get_sym(lib, "add");
  g_bench = (bench_fn)get_sym(lib, "bench_sum");
  if (!g_version || !g_name || !g_run || !g_add || !g_bench) {
    napi_throw_error(env, NULL, "ABI plugin is missing required symbols");
    return exports;
  }

  napi_value fn;
  napi_create_function(env, "name", NAPI_AUTO_LENGTH, Name, NULL, &fn);
  napi_set_named_property(env, exports, "name", fn);
  napi_create_function(env, "interfaceVersion", NAPI_AUTO_LENGTH, Version, NULL, &fn);
  napi_set_named_property(env, exports, "interfaceVersion", fn);
  napi_create_function(env, "add", NAPI_AUTO_LENGTH, Add, NULL, &fn);
  napi_set_named_property(env, exports, "add", fn);
  napi_create_function(env, "benchSum", NAPI_AUTO_LENGTH, BenchSum, NULL, &fn);
  napi_set_named_property(env, exports, "benchSum", fn);
  napi_create_function(env, "run", NAPI_AUTO_LENGTH, Run, NULL, &fn);
  napi_set_named_property(env, exports, "run", fn);
  return exports;
}