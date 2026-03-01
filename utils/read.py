# utils/read.py

def run(ctx, args):
    if len(args) != 1:
        return "Error: read requires 1 argument: <relative_path>"
    path = ctx.validate_path(args[0])
    try:
        # Read with utf-8 and ignore errors to handle any non-utf8 bytes
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # Replace any remaining non-ASCII characters with '?' to avoid encoding errors when printing
        content = content.encode('GBK', errors='replace').decode('GBK')
        return content
    except Exception as e:
        return f"Error: cannot read file {args[0]} - {e}"