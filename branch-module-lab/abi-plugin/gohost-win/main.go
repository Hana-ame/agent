//go:build windows

// Windows 侧 Go 消费者：不需要 cgo，用 syscall.NewLazyDLL 直接加载 .dll。
// 交叉编译验证：GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build
package main

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	greetDLL = syscall.NewLazyDLL("greet.dll")
	procAdd  = greetDLL.NewProc("add")
	procRun  = greetDLL.NewProc("plugin_run")
	procName = greetDLL.NewProc("plugin_name")
)

func cString(p uintptr) string {
	if p == 0 {
		return ""
	}
	var out []byte
	for i := 0; ; i++ {
		c := *(*byte)(unsafe.Pointer(p + uintptr(i)))
		if c == 0 {
			break
		}
		out = append(out, c)
	}
	return string(out)
}

func main() {
	r, _, _ := procAdd.Call(1, 2)
	fmt.Println("add(1,2) =", int32(r))

	p, _, _ := procName.Call()
	fmt.Println("name =", cString(p))

	in := append([]byte("world"), 0)
	out := make([]byte, 512)
	rc, _, _ := procRun.Call(
		uintptr(unsafe.Pointer(&in[0])),
		uintptr(unsafe.Pointer(&out[0])),
		uintptr(len(out)),
	)
	if rc != 0 {
		panic("plugin_run failed")
	}
	n := 0
	for out[n] != 0 {
		n++
	}
	fmt.Println("run(\"world\") =", string(out[:n]))
}