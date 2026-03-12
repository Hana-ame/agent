"# [START] TOOL-APPLY-SNIPPET-BLOCK
# version: 0.0.14
# 描述：从响应文件读取代码段块并更新文件中的代码段（根据文件类型自动选择注释符）

\"\"\"
apply_snippet_block - 从响应文件读取代码段块并更新文件中的代码段

用法：
 py utils.py apply_snippet_block # 默认从 THIS_RESPONSE.txt 读取
 py utils.py apply_snippet_block last # 从 LAST_RESPONSE.txt 读取
 py utils.py apply_snippet_block this # 从 THIS_RESPONSE.txt 读取

响应文件（LAST_RESPONSE.txt 或 THIS_RESPONSE.txt）中需包含按以下格式组织的多个代码段块：

 === 相对路径 ===
 # [START] 代码段名称
 代码段内容
 # [END] 代码段名称
 === end of 相对路径 ===

每个块以 \"=== 路径 ===\" 开始，以 \"=== end of 路径 ===\" 结束。
工具会自动提取代码段并更新到对应文件的指定代码段中。
如果文件不存在，会自动创建并添加代码段。

输出格式：
 - 成功时：每个成功更新的文件返回一行：
 {rel_path} 更新成功：成功：代码段 '{name}' 已写入 {rel_path}
 - 失败时：
 * 整体性错误（如无法读取响应文件、格式错误）返回：
 === apply_snippet_block ===
 错误：具体错误信息
 === end of apply_snippet_block ===
 * 单个文件处理失败（如未找到代码段标记、写入失败）返回针对该文件的错误块：
 === {rel_path} ===
 错误：具体错误信息
 === end of {rel_path} ===
 - 多个结果之间用两个换行分隔。
\"\"\"

import os
import re
from . import snippet

PREVIEW_LENGTH = 250

def _handle_error(subcmd: str, msg: str) -> str:
 return f\"=== {subcmd} ===\
错误：{msg}\
=== end of {subcmd} ===\"

# 用于从响应文件内容中提取代码段名称和内容的函数
def _extract_snippet_from_content(content: str, file_path: str):
 \"\"\"从内容中提取代码段，返回 (name, snippet_content) 或 None，根据 file_path 决定注释符\"\"\"
 pattern = snippet._build_snippet_pattern(file_path) # 注意：_build_snippet_pattern 是内部函数
 match = pattern.search(content)
 if match:
 return match.group(1), match.group(2)
 return None, None

def run(ctx, args):
 # 解析参数，决定使用哪个响应文件
 if args and args[0] in ('this', 'last'):
 which = args[0]
 else:
 which = 'this' # 默认 this

 if which == 'this':
 response_file = os.path.join(ctx.root_path, \".agent\", \"THIS_RESPONSE.txt\")
 else: # 'last'
 response_file = os.path.join(ctx.root_path, \".agent\", \"LAST_RESPONSE.txt\")

 try:
 with open(response_file, \"r\", encoding=\"utf-8\") as f:
 lines = f.readlines()
 except Exception as e:
 return _handle_error(\"apply_snippet_block\", f\"无法读取 {os.path.basename(response_file)} - {e}\")

 blocks = []
 i = 0
 while i < len(lines):
 line = lines[i].rstrip('\
')
 start_match = re.match(r'^=== (.+) ===$', line)
 if start_match:
 rel_path = start_match.group(1).strip()
 i += 1
 content_lines = []
 while i < len(lines):
 current_line = lines[i].rstrip('\
')
 end_match = re.match(r'^=== end of (.+) ===$', current_line)
 if end_match and end_match.group(1).strip() == rel_path:
 break
 content_lines.append(lines[i])
 i += 1
 if i < len(lines) and re.match(r'^=== end of .+ ===$', lines[i].rstrip('\
')):
 i += 1
 file_content = ''.join(content_lines)
 blocks.append((rel_path, file_content))
 else:
 i += 1

 if not blocks:
 response = \"\
\".join(lines)
 if len(response) > PREVIEW_LENGTH * 2:
 msg = f\"{os.path.basename(response_file)} 格式错误（格式：=== 路径 === ... === end of 路径 ===\
预览:\
{response[:PREVIEW_LENGTH]}...(中间省略)...{response[-PREVIEW_LENGTH:]}\"
 else:
 msg = f\"{os.path.basename(response_file)} 格式错误（格式：=== 路径 === ... === end of 路径 ===\
预览:\
{response}\"
 return _handle_error(\"apply_snippet_block\", msg)

 results = []
 for rel_path, file_content in blocks:
 full_path = os.path.join(ctx.root_path, rel_path)
 # 从文件内容中提取代码段，传递文件路径以确定注释符
 name, snippet_content = _extract_snippet_from_content(file_content, full_path)
 if name is None:
 results.append(_handle_error(rel_path, \"内容中未找到有效的代码段标记\"))
 continue

 # 调用 snippet 工具的 set 命令
 snippet_args = ['set', rel_path, name, '--content', snippet_content]
 snippet_result = snippet.run(ctx, snippet_args)

 # 如果 snippet.run 返回的是错误块，提取错误信息
 if snippet_result.startswith(\"=== snippet set ===\"):
 error_lines = snippet_result.splitlines()
 for line in error_lines:
 if line.startswith(\"错误：\"):
 error_msg = line[3:].strip()
 results.append(_handle_error(rel_path, error_msg))
 break
 else:
 results.append(snippet_result)
 else:
 results.append(f\"{rel_path} 更新成功：{snippet_result}\")

 return \"\
\
\".join(results)

# [END] TOOL-APPLY-SNIPPET-BLOCK
"