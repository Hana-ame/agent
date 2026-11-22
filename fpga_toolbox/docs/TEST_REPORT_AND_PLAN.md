# 全工具测试结果与待实机验证方案（v0.5.0）

## 一、已完成实测的工具与产物汇总

测试环境：Linux 沙箱 · Python 3.11.2 · PyYAML 6.0 · `stdbuf` 可用（离线环境，无 Verilator / slang / cocotb / ruff 二进制）。

| 工具 / 产物 | 测试方式 | 结果 | 实测关键输出 |
|---|---|---|---|
| `index_repo` | 实测（`examples/axi_apb_ref`） | **PASS** | 完整索引 40 个 SV/V 模块 |
| `resolve_deps` | 实测 | **PASS** | 正确计算 `axi_apb_bridge` 依赖闭包 |
| `copy_sandbox` | 实测 | **PASS** | 隔离复制 + 拒绝工程根误删保护生效 |
| `extract_interfaces` | 实测（内置解析器） | **PASS** | `axi_apb_bridge.sv` 同时抽出 `axi4_lite(16)` + `apb(8)`；`axi_apb_bridge_sv.sv` 识别 SV `interface/modport` 并输出 `W012` |
| `validate_contract` | 实测 | **PASS** | 直接通过 `examples/axi_apb_ref/axi/regfile_spec.json`（12 个寄存器 + APB），拦截 `E013` 位域重叠与 `E014` 超宽 |
| `emit_testbench` | 实测 | **PASS** | 生成 4 份产物（wrapper/SVA/bind/cocotb），生成的 cocotb Python 编译通过 |
| `normalize_sim_result` | 实测 | **PASS** | 校验 JUnit XML、Verilator 多套件日志身份核对，漏跑套件必报 FAIL、非法 JSON 必拒 |
| `render_visual_report` | 实测 | **PASS** | 输出 SVG + HTML 看板，`consistency_ok=True`，渲染覆盖率与变异摘要 |
| `lint_tb` | 实测 | **PASS** | 检出故障注入脚本中的 `C1`、`R2`、`S1`、`S3` 违规 |
| `run_mutation` | 实测 | **PASS** | 100% 变异杀死、`try/finally` 原文件完好还原、残留变异触发 `E030` 拦截 |
| `audit_harness` | 实测 | **PASS** | 4/4 故障注入自审计全绿（`A1–A4`） |
| `run_sim` | **部分实测（门禁）** | **PASS（门禁） / 待实机仿真** | 缺失 Verilator 时正确触发 `E020` 拒绝静默通过；真实仿真需外部 EDA 环境（见第二节） |
| `tests/test_regression.py` | 全量运行 | **PASS** | `ALL CHECKS PASSED` |
| `tests/test_fusion_and_cross_repo.py` | 全量运行 | **PASS** | `ALL FUSION CHECKS PASSED` |
| `web_tools/protocol-editor.html` | `check_web_app.py` + 无头渲染 | **PASS** | 0 errors, 0 warnings，桌面端渲染正常 |
| `web_tools/repo-comparison.html` | 静态自包含检查 | **PASS** | 0 `<script>`，图片已 base64 内联 |

---

## 二、当前环境无法实测项的完整测试方案

以下 5 类场景依赖外部二进制工具或联网环境，请在目标开发机或 GitHub Actions CI 按如下步骤验证：

### 1. `run_sim` 真实 Verilator ≥ 5.040 + slang 仿真
- **前置准备**：在 `examples/axi_apb_ref/` 下执行 `bash setup_env.sh --check`；若不满足版本要求，执行 `bash setup_env.sh` 幂等编译安装 Verilator ≥ 5.040 至 `~/.local/verilator`，并安装 `slang`。
- **执行命令**：
  ```bash
  python3 -m fpga_toolbox run_sim \
    --sim-script examples/axi_apb_ref/sim.sh \
    --suite all \
    --expected-suites '["counter","counter8","axi","sv","gaps","dual","stress","covgaps","monitor","regfile","regfilegen","vip"]' \
    --require-slang
  ```
- **验收标准**：返回 `"ok": true`，`suite_identity.all_verified == true`，且任意单套件注入缺陷时返回非零退出码与 `E022/E023`。

### 2. `extract_interfaces` 的 `slang --ast-json` 后端验证
- **执行命令**：
  ```bash
  python3 -m fpga_toolbox extract_interfaces \
    --rtl-file examples/axi_apb_ref/axi/sv/axi_apb_bridge_sv.sv \
    --parser-backend slang
  ```
- **验收标准**：`parser_backend == "slang"`，不再输出 `W012`，正确识别 `axi` 与 `apb` 接口端口与参数。

### 3. `emit_testbench` 生成的 cocotb + Verilator 实机仿真
- **前置准备**：`pip install cocotb cocotbext-axi` + Verilator ≥ 5.040。
- **执行步骤**：对 `emit_testbench` 产出的 `*_axil_wrap.sv`、`*_axil_sva_checker.sv` 与 `test_*_axil.py` 运行 `SIM=verilator pytest` 或 cocotb Makefile，将生成的 `test_results_*.json` 与 SVA 日志喂入 `normalize_sim_result`。
- **验收标准**：复位值、RW 写回读、RO 不可变、非法地址响应与随机反压压测全绿，`normalize_sim_result` 返回 `overall_status == "PASS"`。

### 4. `run_mutation` 对 `examples/axi_apb_ref/` 的 33 项完整 RTL 变异测试
- **执行步骤**：在 `copy_sandbox` 隔离副本中，把 `examples/axi_apb_ref/axi/mutate.py` 的 33 条 `MUTATIONS` 导出为 JSON 规格并调用 `python3 -m fpga_toolbox run_mutation --workdir <sandbox> --mutation-spec muts.json --test-cmd '["bash","sim.sh","all"]' --min-score 80`。
- **验收标准**：复现 33/33（100%）变异得分；中途 `Ctrl+C` 中断后重新运行，确认 `E030` 残留保护生效。

### 5. GitHub Actions CI (`ruff` + 多平台矩阵 + 打包发版) 与 TLA+ TLC 验证
- **CI 验证**：推送到 GitHub `main` 或提 PR，确认 `.github/workflows/ci.yml` 的 `lint`（`ruff check fpga_toolbox tests` + Python 3.10 AST）、`test`（Ubuntu 3.10/3.11/3.13 + Windows 3.13）和 `build`（wheel/sdist 安装验证）全部绿灯。
- **TLA+ 验证**：用浏览器打开 `web_tools/protocol-editor.html`，在第 4 页导出 `.tla` 模型，放入 VS Code TLA+ 插件配置 `MaxRespCycles = 4` 并检查不变量 `SafetyInvariant` 与活性 `EventuallyResponds`，确认 TLC 状态空间穷举通过。
