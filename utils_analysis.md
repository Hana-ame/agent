# utils 工具可用性分析报告

## 生成时间：$(py utils.py time --format '%Y-%m-%d %H:%M:%S')

## 工具列表及状态

| 工具名 | 状态 | 说明 |
|--------|------|------|
| append | ✅ 可用 |  |
| cat | ✅ 可用 |  |
| delete | ✅ 可用 |  |
| git | ✅ 可用 |  |
| help | ✅ 可用 |  |
| ls | ✅ 可用 |  |
| memory | ✅ 可用 | SQLite 记忆存储，功能完整 |
| mv | ✅ 可用 |  |
| pause | ✅ 可用 |  |
| replace | ✅ 可用 |  |
| search | ✅ 可用 | 搜索内容（具体功能待完善） |
| snippet | ✅ 可用 | 管理代码段（list/get/set/delete），功能完整 |
| time | ✅ 可用 | 显示时间/计时器（缺少帮助文档，但功能正常） |
| write | ✅ 可用 | 覆盖写入文件（版本 002，普通写入，缺少帮助文档） |
| write_multiple | ✅ 可用 | 批量写入多个文件（需正确格式的响应文件） |
| _command_executor | ⚠️ 辅助模块 | 已重命名，不应直接调用 |
| _file_utils | ⚠️ 辅助模块 | 已重命名，不应直接调用 |

## 不可用工具说明

- 无不可用工具。所有列出的工具均已实现 `run` 函数。

## 修改方向建议

1. 为 `write` 和 `time` 等工具补充详细的帮助文档（修改模块内的文档字符串）。
2. 更新 `agent.py` 和 `agent_v2.py` 中的导入，将 `from file_utils import read_system_prompt` 改为 `from utils._file_utils import read_system_prompt`。
3. 如需增强版写入工具（支持 `--snippet`），请参考新创建的 `write_snippet.py` 工具。