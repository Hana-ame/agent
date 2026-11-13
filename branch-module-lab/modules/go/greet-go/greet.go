package main

import "example.com/bml/go/sdk"

type mod struct{}

func (mod) Name() string { return "greet-go" }

func (mod) Describe() string { return "Go 插件模块：生成问候语（v3）" }

func (mod) Run(payload string) (string, error) {
	return "hello, " + payload + " (go v3)", nil
}

// New 是宿主约定的导出符号；必须返回 sdk.Module。
func New() sdk.Module { return mod{} }
