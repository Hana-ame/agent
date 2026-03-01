# utils.py 工具集使用说明

本文档介绍了 `utils.py` 工具集中各工具的功能及使用方法。所有工具通过 `py utils.py <工具名> [参数...]` 调用，路径均相对于 `.env` 中定义的 `ROOT_PATH`。

## 可用工具列表

### 1. read
**用途**：读取指定文件的内容并输出到终端。  
**用法**：`py utils.py read <相对路径>`  
**示例**：`py utils.py read config.json`

### 2. write
**用途**：将上一次 LLM 响应（即 `LAST_RESPONSE.txt` 的内容）写入指定文件。  
**用法**：`py utils.py write <相对路径>`  
**注意**：需先通过纯文本输出内容，由 Agent 自动保存到 `LAST_RESPONSE.txt` 后再执行此命令。

### 3. append
**用途**：将指定内容追加到文件末尾。若文件不存在则自动创建。  
**用法**：`py utils.py append <相对路径> <内容>`  
**示例**：`py utils.py append logs/access.log "2025-03-01 12:00:00 new entry"`

### 4. replace
**用途**：将文件中的旧文本替换为新文本（简单字符串替换）。  
**用法**：`py utils.py replace <相对路径> <旧文本> <新文本>`  
**示例**：`py utils.py replace src/main.py "DEBUG=True" "DEBUG=False"`

### 5. list
**用途**：列出指定目录下的所有文件和文件夹。  
**用法**：`py utils.py list <相对路径>`  
**示例**：`py utils.py list src`

### 6. delete
**用途**：删除指定文件。  
**用法**：`py utils.py delete <相对路径>`  
**示例**：`py utils.py delete temp.txt`

### 7. pause
**用途**：在根目录创建 `.pause` 文件，触发 Agent 暂停循环，等待人工干预。  
**用法**：`py utils.py pause`  
**输出**：以 `PAUSED:` 开头的提示信息。

### 8. resume
**用途**：删除根目录的 `.pause` 文件，使 Agent 恢复运行。  
**用法**：`py utils.py resume`

### 9. write_multiple
**用途**：从 `LAST_RESPONSE.txt` 中解析多个文件块并批量写入。文件块格式如下：
```
=== 相对路径1 ===
文件内容（可包含换行）
===
=== 相对路径2 ===
文件内容2
===
```
**用法**：`py utils.py write_multiple`

### 10. git
**用途**：执行常见的 Git 操作（如 status, add, commit, push 等）。  
**用法**：`py utils.py git <git命令> [参数...]`  
**示例**：  
- `py utils.py git status`  
- `py utils.py git add .`  
- `py utils.py git commit -m "update"`  
- `py utils.py git push origin main`

## 注意事项
- 所有路径均受根目录限制，无法访问上层目录（`../`）。
- 命令中的参数若包含空格，需用引号括起。
- 工具执行失败时会返回错误信息，请根据提示调整指令。
- 可通过自定义工具放入 `utils/` 目录扩展功能（需实现 `run(ctx, args)` 函数）。

更多详情请参考各工具源码或联系管理员。