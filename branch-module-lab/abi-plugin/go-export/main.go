// 反向演示：Go 也能用 -buildmode=c-shared 编译出 DLL，供其它语言按 C ABI 调用。
// 交叉编译：GOOS=windows GOARCH=amd64 CGO_ENABLED=1 CC=x86_64-w64-mingw32-gcc \
//            go build -buildmode=c-shared -o build/greet-go.dll .
package main

/*
#include <stdlib.h>
*/
import "C"

//export PluginInterfaceVersion
func PluginInterfaceVersion() C.int { return 1 }

//export PluginName
func PluginName() *C.char { return C.CString("greet-go-dll") }

//export Add
func Add(a, b C.int) C.int { return a + b }

func main() {}