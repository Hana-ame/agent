# AI写的README.md

## 📌 项目简介

`kimi.py` 是一个高级的打字机效果 HTTP 客户端，专为调用 OpenAI 兼容 API 设计。它支持文件附件（包括图片和文本）、参数安全处理、流式响应、以及优雅的错误回退机制，确保在遇到不支持的参数或网络问题时不会崩溃。

---

## 🚀 核心功能

### 🧠 打字机式输出
- 以打字机效果逐字显示响应内容，增强交互体验
- 支持流式（`stream`）和非流式（`non-stream`）响应处理

### 📁 文件支持
- 支持 `@filename` 语法直接附加文件
- 支持常见图像格式（PNG/JPG/GIF/WEBP/BMP）
- 自动检测文本编码（UTF-8, GBK, Latin-1 等）

### 🛡️ 参数安全处理
- 零默认策略：避免意外使用不支持的参数
- 优雅回退：CLI 参数优先于配置文件，配置文件优先于环境变量
- 自动清理无效参数（如 `None` 值、空字符串）

### 📁 配置管理
- 支持 `config.json` 作为请求体配置
- 支持 `profiles.json` 或 `.env` 文件管理 API 端点和密钥
- 可通过 `--profile` 指定预设配置

### 📤 结果保存
- 自动保存为 Markdown (`*.md`) 和 JSON (`*.json`) 格式
- 包含完整对话历史、思考过程、统计信息

---

## 📦 安装依赖

```bash
pip install requests python-dotenv
```

---

## 🧩 使用方式

```bash
python kimi.py "Your prompt here"
```

### 基础用法
```bash
# 不带参数运行（使用默认配置）
python kimi.py "Hello, how are you?"

# 附加文件（支持图片和文本）
python kimi.py @image.png "Describe this image"
python kimi.py @code.py "Explain this code"
```

### 高级配置
```bash
# 指定配置文件和 profile
python kimi.py -p my_profile -c custom_config.json "Custom prompt"

# 继续对话（使用之前的 JSON 输出）
python kimi.py --context conversation.json "Continue the conversation"

# 禁用流式输出
python kimi.py --no-stream "Generate a long response"

# 覆盖参数（如温度、最大 tokens）
python kimi.py --temperature 0.5 --max-tokens 2000 "Creative writing"
```

---

## 📋 参数说明

| 参数 | 说明 |
|------|------|
| `--profile` / `-p` | 使用指定 profile（默认 `default`） |
| `--endpoint` / `-e` | 覆盖 API 端点 |
| `--api-key` / `-k` | 覆盖 API 密钥 |
| `--config` / `-c` | 指定请求体配置文件（默认 `config.json`） |
| `--context` | 使用对话历史 JSON 文件继续对话 |
| `--no-stream` | 禁用流式输出 |
| `--enable-thinking` / `--no-thinking` | 启用/禁用思考过程输出 |
| `--output` / `-o` | 指定输出文件名（默认自动命名） |

---

## 📜 示例配置文件

`config.json` 示例：
```json
{
  "model": "qwen-plus",
  "messages": [
    {"role": "user", "content": "What's the capital of France?"}
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

`profiles.json` 示例：
```json
{
  "default": {
    "endpoint": "https://api.example.com/v1/completions",
    "api_key": "your_api_key_here"
  },
  "deepseek": {
    "endpoint": "https://api.deepseek.com/v1/completions",
    "api_key": "deepseek_api_key"
  }
}
```

---

## 📌 注意事项

1. **Token 限制**  
   - 预估 token 数（每 3 字符 ≈ 1 token）  
   - 注意 OpenAI 兼容 API 的上下文长度限制（如 32k/128k tokens）

2. **文件大小**  
   - 支持大文件上传，但需注意 API 的文件大小限制

3. **参数冲突**  
   - CLI 参数会安全覆盖配置文件中的参数  
   - 不支持的参数（如 `temperature` 与特定模型冲突）会被自动移除

4. **安全建议**  
   - 不要将 API 密钥硬编码在脚本中  
   - 使用 `.env` 文件或 profiles.json 管理敏感信息

---

## 📝 输出示例

```markdown
# AI Conversation Log
**Time**: 2023-10-05 14:30:00  
**Model**: qwen-plus  
**Duration**: 1.23s  

## Request Configuration
```json
{
  "model": "qwen-plus",
  "temperature": 0.7,
  "max_tokens": 100
}
```

## Conversation History

### USER
Hello, how are you?

## 💭 Thinking Process
<details>
<summary>Click to expand (123 chars)</summary>
This is the thinking process...
</details>

## ✨ Response
This is the final response...

## 📊 Statistics
- **Finish Reason**: stop
- **Tokens**: 150 (Prompt: 50, Completion: 100)
- **Token Rate**: 123.4 tokens/s
