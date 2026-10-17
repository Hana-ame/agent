"""把当前主源改造成「模板形态」：清单/代码块清空、自指槽换成定长标记行、槽宽归零。

模板形态的意义：清单和源码块填进去会改字节数，无法与"槽宽"同时求解；
清空后，模板里唯一的长度变量是 `BOOTSTRAP_SLOT = <N>` 的数字位数（封顶后恒定）。

用法：
    python tools/to_template.py            # 就地转换（幂等）
    python tools/to_template.py --check    # 只报告是否已是模板形态
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = "vea_v6/parts/bootstrap.py"

EMPTY_MANIFEST = '{\n  "spec": {},\n  "files": []\n}'
EMPTY_CODE = '{\n  "_self": "PENDING",\n  "files": []\n}'

MAN_RE = re.compile(r"^FRAME_MANIFEST = json\.loads\(r'''.*?'''\)", re.S | re.M)
CODE_RE = re.compile(r"^FRAME_CODE = json\.loads\(r'''.*?'''\)", re.S | re.M)
SELF_RE = re.compile(r'^SELF_FILL = "F+"$', re.M)
SLOT_RE = re.compile(r"^BOOTSTRAP_SLOT = \d+.*$", re.M)

SELF_LINE = 'SELF_FILL = "' + "F" * 64 + '"'
SLOT_LINE = "BOOTSTRAP_SLOT = 0  # 自指 payload 的 base64 文本长度（由 tools/fixpoint_v3.py 填入）"


def to_template(src: str) -> str:
    out, n1 = MAN_RE.subn(lambda _m: "FRAME_MANIFEST = json.loads(r'''" + EMPTY_MANIFEST + "''')", src, count=1)
    if n1 != 1:
        raise ValueError("缺少 FRAME_MANIFEST 块")
    out, n2 = CODE_RE.subn(lambda _m: "FRAME_CODE = json.loads(r'''" + EMPTY_CODE + "''')", out, count=1)
    if n2 != 1:
        raise ValueError("缺少 FRAME_CODE 块")
    if SELF_RE.search(out):
        out = SELF_RE.sub(lambda _m: SELF_LINE, out, count=1)
    else:
        # 首次转换：在 BOOTSTRAP_SLOT 之前插入 SELF_FILL 行
        out = SLOT_RE.sub(lambda _m: SELF_LINE + "\n" + SLOT_LINE, out, count=1)
        if not SELF_RE.search(out):
            raise ValueError("SELF_FILL 未能插入")
    out = SLOT_RE.sub(lambda _m: SLOT_LINE, out, count=1)
    return out


def main(argv: list) -> int:
    path = ROOT / MAIN
    src = path.read_text(encoding="utf-8")
    is_tpl = bool(SELF_RE.search(src)) and '  "spec": {},\n  "files": []' in src
    if "--check" in argv:
        print("OK  模板形态" if is_tpl else "NO  不是模板形态")
        return 0 if is_tpl else 1
    if is_tpl and "--force" not in argv:
        print("已是模板形态，未改动（--force 可重写）")
        return 0
    out = to_template(src)
    path.write_text(out, encoding="utf-8")
    print(f"已转为模板形态：{len(out.encode('utf-8'))} 字节")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
