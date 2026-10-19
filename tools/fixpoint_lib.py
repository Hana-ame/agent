"""⚠️ 已废弃：早期固定点求解库，与当前双层不动点方案不兼容。现行入口 `tools/fixpoint_v3.py`。"""
import ast, base64, gzip, hashlib, json, re, sys
sys.path.insert(0, '.')

SELF_FILL = "SELFB64FILL"
MAIN = "vea_v6/parts/bootstrap.py"

def signature(path, source, spec_files=("vea_v6/__init__.py",)):
    if path in spec_files:
        m = re.search(r"^__all__\s*=\s*\[(.*?)\]", source, re.S | re.M)
        return re.findall(r'"([^"]+)"', m.group(1))
    tree = ast.parse(source)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                out.append("def %s(%s)" % (node.name, ", ".join(a.arg for a in node.args.args)))
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                out.append("class %s(%s)" % (node.name, ", ".join(ast.unparse(b) for b in node.bases)))
    return out

def b64of(raw):
    return base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode("ascii")

MAN_RE = re.compile(r"^FRAME_MANIFEST = json\.loads\(r'''.*?'''\)", re.S | re.M)
CODE_RE = re.compile(r"^FRAME_CODE = json\.loads\(r'''.*?'''\)", re.S | re.M)

def render(source, manifest_text, code_text, pad=0):
    out, n1 = MAN_RE.subn(lambda m: "FRAME_MANIFEST = json.loads(r'''" + manifest_text + "''')", source, count=1)
    assert n1 == 1
    out, n2 = CODE_RE.subn(lambda m: "FRAME_CODE = json.loads(r'''" + code_text + "''')", out, count=1)
    assert n2 == 1
    return out.replace(SELF_FILL, "F" * pad)

def probe(src):
    ns = {"__name__": "_probe"}
    exec(compile(src, "<probe>", "exec"), ns)
    return ns
