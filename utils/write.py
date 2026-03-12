
# [START] TOOL-WRITE
# version: 002
# 描述：覆盖写入文件
import os

def run(ctx, args):
    if len(args) < 2:
        return "Error: write 需要 2 个参数：<路径> <内容...>"
    
    rel_path = args[0]
    # 将剩余参数重新拼接为内容 (处理空格)
    content = " ".join(args[1:])
    
    try:
        full_path = ctx.validate_path(rel_path)
        # 自动创建父目录
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"Success: 已写入 {rel_path} (长度: {len(content)})"
    except Exception as e:
        return f"Error: 写入失败 - {str(e)}"
# [END] TOOL-WRITE\n# [START] TOOL-WRITE\n"\"\"\"""write - 覆盖写入文件""""用法："" py utils.py write <路径> <内容...>""\"\"\"""# version: 002""# 描述：覆盖写入文件""""import os""""def run(ctx, args):"" if len(args) < 2:"" return \"Error: write 需要 2 个参数：<路径> <内容...>\""" "" rel_path = args[0]"" content = \" \".join(args[1:])"" "" try:"" full_path = ctx.validate_path(rel_path)"" os.makedirs(os.path.dirname(full_path), exist_ok=True)"" with open(full_path, 'w', encoding='utf-8') as f:"" f.write(content)"" return f\"Success: 已写入 {rel_path} (长度: {len(content)})\""" except Exception as e:"" return f\"Error: 写入失败 - {str(e)}\""\n# [END] TOOL-WRITE\n