// Package sdk 定义宿主与 Go 插件模块之间的稳定契约（interface v1）。
//
// 关键点：接口必须放在主仓库的共享包里。宿主和插件都 import 这个包，
// plugin.Open 之后用 func() sdk.Module 做类型断言，Go 才认为两边是同一个类型。
package sdk

// Module 是所有 Go 模块必须实现的接口。
//
//	New() sdk.Module —— 模块包（package main）必须导出的符号。
//	                   plugin.Lookup("New") 拿到 func() sdk.Module 后调用即可。
type Module interface {
	// Name 模块名，必须与 modules/<lang>/<name> 目录名一致。
	Name() string
	// Describe 一句话说明模块能力。
	Describe() string
	// Run 模块的单一功能入口。demo 用 string；真实项目可换成结构体/JSON。
	Run(payload string) (string, error)
}
