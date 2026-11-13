// Command host 是 Go 语言的模块宿主（runtime host）。
//
// 它只做三件事：
//  1. 扫描 modules/go/*/module.so（由 sync.py 从 feature 分支合并、构建而来）
//  2. 用标准库 plugin 包把每个 .so 当模块加载，类型断言成 sdk.Module
//  3. 调用统一接口 Name/Describe/Run，并（--watch 时）检测 .so 变化后 re-exec 自更新
//
// 使用：
//
//	go build -o .bin/go-host ./go/host
//	./.bin/go-host --root .
//	./.bin/go-host --root . --watch
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"plugin"
	"sort"
	"syscall"
	"time"

	"example.com/bml/go/sdk"
)

type lockModule struct {
	Name     string `json:"name"`
	Language string `json:"language"`
	Branch   string `json:"branch"`
	Commit   string `json:"commit"`
	Version  string `json:"version"`
	Artifact string `json:"artifact"`
}

type lockFile struct {
	Modules []lockModule `json:"modules"`
}

func main() {
	root := flag.String("root", ".", "仓库根目录（应在 branch-module-lab 根目录运行）")
	watch := flag.Bool("watch", false, "监听 module.so 变化，自动重新加载（re-exec）")
	payload := flag.String("payload", "world", "传给模块 Run 的输入")
	flag.Parse()

	absRoot, err := filepath.Abs(*root)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[go] 无法解析 --root: %v\n", err)
		os.Exit(1)
	}

	if *watch {
		fmt.Println("[go] watch 模式：module.so 变化后会自动 re-exec 宿主（Go plugin 无法卸载）")
	}
	loadAndRun(absRoot, *payload)

	if *watch {
		watchAndReexec(absRoot)
	}
}

func modulesDir(root string) string {
	return filepath.Join(root, "modules", "go")
}

func discover(root string) []string {
	matches, _ := filepath.Glob(filepath.Join(modulesDir(root), "*", "module.so"))
	sort.Strings(matches)
	return matches
}

func loadLock(root string) map[string]lockModule {
	out := map[string]lockModule{}
	raw, err := os.ReadFile(filepath.Join(root, "modules.lock.json"))
	if err != nil {
		return out
	}
	var lf lockFile
	if json.Unmarshal(raw, &lf) != nil {
		return out
	}
	for _, m := range lf.Modules {
		out[m.Name] = m
	}
	return out
}

func short(sha string) string {
	if len(sha) >= 7 {
		return sha[:7]
	}
	if sha == "" {
		return "?"
	}
	return sha
}

func loadAndRun(root, payload string) {
	files := discover(root)
	if len(files) == 0 {
		fmt.Println("[go] 没有发现 Go 模块（先运行：python3 sync.py once）")
		return
	}
	lock := loadLock(root)
	fmt.Printf("[go] 发现 %d 个 Go 模块\n", len(files))
	for _, path := range files {
		name := filepath.Base(filepath.Dir(path))
		info := lock[name]
		ver := info.Version
		if ver == "" {
			ver = "?"
		}

		p, err := plugin.Open(path)
		if err != nil {
			fmt.Printf("[go] ✗ %s 加载失败: %v\n", name, err)
			continue
		}
		sym, err := p.Lookup("New")
		if err != nil {
			fmt.Printf("[go] ✗ %s 缺少导出符号 New: %v\n", name, err)
			continue
		}
		newFn, ok := sym.(func() sdk.Module)
		if !ok {
			fmt.Printf("[go] ✗ %s 的 New 不是 func() sdk.Module\n", name)
			continue
		}
		mod := newFn()
		fmt.Printf("[go] %s v%s commit=%s | %s\n", mod.Name(), ver, short(info.Commit), mod.Describe())
		out, err := mod.Run(payload)
		if err != nil {
			fmt.Printf("[go]   run(%q) 出错: %v\n", payload, err)
			continue
		}
		fmt.Printf("[go]   run(%q) => %s\n", payload, out)
	}
}

// snapshot 记录所有 module.so 的 mtime+size，用于判断同步器是否重建过插件。
func snapshot(root string) map[string]int64 {
	out := map[string]int64{}
	for _, f := range discover(root) {
		st, err := os.Stat(f)
		if err != nil {
			continue
		}
		out[f] = st.ModTime().UnixNano() ^ st.Size()
	}
	return out
}

func changes(a, b map[string]int64) bool {
	if len(a) != len(b) {
		return true
	}
	for k, v := range a {
		if b[k] != v {
			return true
		}
	}
	return false
}

func watchAndReexec(root string) {
	prev := snapshot(root)
	for {
		time.Sleep(500 * time.Millisecond)
		next := snapshot(root)
		if changes(prev, next) {
			fmt.Println("[go] 检测到 module.so 更新 → re-exec 宿主以加载新插件")
			exe, err := os.Executable()
			if err != nil {
				fmt.Printf("[go] re-exec 失败: %v\n", err)
				return
			}
			if err := syscall.Exec(exe, os.Args, os.Environ()); err != nil {
				fmt.Printf("[go] re-exec 失败: %v\n", err)
				return
			}
			return
		}
	}
}
