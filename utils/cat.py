
# [START] TOOL-READ
# version: 002
# 描述：读取文件内容
import os

def run(ctx, args):
    if not args:
        return "Error: read 需要至少一个文件路径"
    
    results = []
    for rel_path in args:
        try:
            full_path = ctx.validate_path(rel_path)
            if not os.path.exists(full_path):
                results.append(f"=== {rel_path} ===\nError: 文件不存在")
                continue
                
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            # 添加文件名标记，方便 AI 区分多个文件
            results.append(f"=== {rel_path} ===\n{content}\n=== end of {rel_path} ===")
        except Exception as e:
            results.append(f"=== {rel_path} ===\nError: {str(e)}")
            
    return "\n\n".join(results)
# [END] TOOL-READ