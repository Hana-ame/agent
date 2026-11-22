# fpga_toolbox v0.5.0 — 融合版 AI Agent FPGA 验证工具箱

将 **`fpga_toolbox`** 的 AI Agent / MCP 控制面与 **`agent/verilog`** 沉淀出的硬件验证与反静默通过护栏融为一体：

- **控制面（零 EDA 强依赖，仅 PyYAML）**：12 个原子工具统一走 `ToolResult(ok / tool / data / diagnostics / error)` 信封，自动导出 MCP JSON Schema 与 CLI 子命令。
- **验证与防假通过护栏**：支持 APB 协议分类、SV `interface/modport` 识别与可选 `slang` AST 后端、位域契约校验（兼容 `apb-regfile-spec/v2`）、三套静态反踩坑 Lint（时钟竞争 / AXI 握手 / 静默通过）、参数化变异测试（`try/finally` 强还原 + 残留拦截）、Verilator ≥ 5.040 封装与故障注入自审计。
- **参考靶场（`examples/axi_apb_ref/`）**：完整保留原 `agent/verilog` 的 AXI4-Lite ↔ APB 串行/并发桥、25 个 Verilator 测试套件、VIP/UVM-lite 与 `regfile_spec.json` 作为跨库集成回归基准。

## 12 个原子工具一览

| # | 工具 | 职责 |
|---|---|---|
| 1 | `index_repo` | 解析 RTL 仓库模块、例化关系与头文件索引 |
| 2 | `resolve_deps` | BFS 计算目标模块传递依赖闭包与 DAG 边 |
| 3 | `copy_sandbox` | 带防误删保护与 SHA-256 校验的沙箱隔离复制 |
| 4 | `extract_interfaces` | 抽取 AXI4-Lite / AXI4-Stream / APB / SV `interface` 与时钟复位（支持可选 `slang` AST） |
| 5 | `validate_contract` | 校验 YAML/JSON 边界契约与 `apb-regfile-spec/v2` 位域规则（E000–E014） |
| 6 | `emit_testbench` | 生成 SV wrapper / SVA / bind 与含写回读、反压压测的 cocotb 测试平台 |
| 7 | `normalize_sim_result` | 归一化 cocotb XML/JSON、SVA 日志与 Verilator 多套件日志（含逐套件身份核对） |
| 8 | `render_visual_report` | 渲染独立 SVG 拓扑 + HTML 规则审计看板，内置 HTML 标签与行数一致性自检 |
| 9 | `lint_tb` | 静态反踩坑检查：时钟沿竞争（C1–C2）、AXI 握手死锁（R1–R4）、静默通过（S1–S3） |
| 10 | `run_sim` | 封装 Verilator ≥ 5.040 + slang 仿真、`stdbuf` 日志保活与逐套件身份校验 |
| 11 | `run_mutation` | 参数化 RTL 变异测试，含 `try/finally` 强还原与残留变异拦截（E030） |
| 12 | `audit_harness` | 故障注入式自审计（A1–A4），证明验证护栏出错时不会静默通过 |

## 快速验证与测试

```bash
pip install pyyaml
python3 tests/test_regression.py
python3 tests/test_fusion_and_cross_repo.py
python3 -m fpga_toolbox audit_harness
python3 -m fpga_toolbox list
```
