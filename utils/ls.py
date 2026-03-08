
# [START] TOOL-LIST
# version: 002
# 描述：列出目录文件
import os

def run(ctx, args):
    rel_path = args[0] if args else "."
    
    try:
        full_path = ctx.validate_path(rel_path)
        if not os.path.isdir(full_path):
            return f"Error: {rel_path} 不是一个目录"
            
        items = os.listdir(full_path)
        result = []
        for item in items:
            item_path = os.path.join(full_path, item)
            suffix = "/" if os.path.isdir(item_path) else ""
            result.append(f"{item}{suffix}")
            
        return "\n".join(result) if result else "(空目录)"
    except Exception as e:
        return f"Error: 无法列出目录 - {str(e)}"
# [END] TOOL-LIST