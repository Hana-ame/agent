"""VEA V6 自举模块：框架（整个包）的自生成与自复现。

一句话：**用 DAG 把「框架的 SPEC」变成「框架的源码」，生成的框架能逐字节重建自己**。

## 三样东西

| 概念 | 含义 |
|---|---|
| `FRAMEWORK_SPEC` | 框架规格：名称/不变式/特性/公开 API，是渲染的语义输入 |
| `FRAME_MANIFEST` | 框架清单：SPEC + 每文件的字节数 / b64 长度 / sha256 / 公开签名 |
| `FRAME_CODE` | 框架源码（`{路径, key, code}`）；除本文件外都 gzip+base64 内嵌，本文件用 `BOOTSTRAP_B64` 占位，定长填充在 `SELF_FILL` |

## 固定点：为什么不是 `z(pad) == pad`

朴素做法是把整个包（含本文件）gzip 后嵌进本文件，得 `L = C + 4⌈L/3⌉`，系数 4/3 > 1，**必然发散**。

于是改成"定长填充"：`SELF_FILL` 是一串 `F`，长度 pad 恰好等于自身 payload 的 base64 长度。
但把库清单一并塞进去后还有一层麻烦：**gzip 对"单调递增的重复字节"几乎免费**，
所以 `z(pad)`（渲染结果的 gzip 体积）在真解附近几乎是常数，`z(pad) - pad` 会从 +12 直接跳到 -3，
**跨过零点而不命中**。窗口取证：`python tools/probe_window.py 32120 32230` → 严格不动点「无」。

现在的解法是**两层解耦**：

```
主源  = 代码 + 定长 SELF_FILL 标记 + BOOTSTRAP_SLOT = pad     ← 字节数与 pad 的"值"无关（位数封顶后恒定）
payload = b64(gzip(最终源码))                                  ← 长度恰为 pad
槽内填充 = (pad - len(sha256)) 个 'x' + sha256(渲染后的源码)
```

于是 `z(pad)` 与 pad 无关（是个常数 z0），取 **pad = 最小的素数 ≥ z0** 即可：
`z0 <= pad` 保证"填充 + sha 尾巴"总能填满槽（最多浪费几十字节）。

`assemble()` 用同一套流程给任意文件集补清单并自校验——**它是模型侧唯一写入口**：
模型只交代码，不碰清单、不碰槽宽、不碰 sha。

## 生成链（一次 shot 连出多代）

`create_generation_dag()` 是标准 V6 DAG：

```
input ─┬─► spec_plan ─────┐
       ├─► skeleton_plan ─┼─► skeleton_code ─┐
       └─► flesh_plan ────┘                  └─► frame_code ─► manifest ─► result
```

`skeleton_code` 要 3 个 tag，单条信息都不满足 ⇒ **必然**由预设边合成。
"""
from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from vea_v6.parts.builder import parse_json_stream
from vea_v6.parts.dag import DEFAULT_PRESET_EDGE, DAG, load_models_file
from vea_v6.parts.llm import OpenAICompatibleLLM
from vea_v6.parts.primitives import ContentItem, Edge

__all__ = [
    "FRAMEWORK_SPEC",
    "FRAME_MANIFEST",
    "FRAME_CODE",
    "MAIN_SOURCE",
    "FRAME_FILES",
    "SPEC_FILES",
    "SELF_FILL",
    "BOOTSTRAP_SLOT",
    "sha_of",
    "fill_for",
    "slot_payload",
    "framework_sources",
    "manifest",
    "generate_framework",
    "verify_self_reproduction",
    "verify_fixed_point",
    "render_framework_code",
    "write_framework",
    "signature",
    "assemble",
    "BootstrapResult",
    "bootstrap",
    "create_bootstrap_dag",
    "create_generation_dag",
    "normalize_spec",
    "spec_to_text",
]

MAIN_SOURCE = "vea_v6/parts/bootstrap.py"
SELF_MARK = "BOOTSTRAP_B64"
#: 自指槽的定长填充标记（长度由 BOOTSTRAP_SLOT 决定；模板形态下这里只是一小串 F）
SELF_FILL = "F" * 64


#: 模块自身引用：dataclass 会按 `cls.__module__` 反查 `sys.modules` 来解析注解，
#: 所以本模块**必须**在 `sys.modules` 里注册成自己的名字。
#: `import vea_v6` 时导入系统会自动注册；但 `mod = types.ModuleType(name);
#: exec(src, mod.__dict__)` 这种"内存执行"的加载方式不会，导致 dataclass 拿不到模块而崩。
#: 这里用 `gc.get_referrers(globals())` 反查出正在承载本模块的 ModuleType 对象补注册——
#: 不碰帧、不改正在迭代的容器，只在缺失时写一次。
if __name__ != "__main__":
    import sys as _sys

    if _sys.modules.get(__name__) is None:
        import gc as _gc

        for _ref in _gc.get_referrers(globals()):
            if isinstance(_ref, type(_sys)) and getattr(_ref, "__name__", None) == __name__:
                _sys.modules[__name__] = _ref
                break


def _fmt_fill(n: int) -> str:
    """SELF_FILL 的定义行：用 `"F" * n` 表达，长度由 n 决定（不写死长串）。"""
    return 'SELF_FILL = "F" * ' + str(int(n))
#: 自指 payload 的 base64 文本长度（不动点；0 表示尚未固化）
BOOTSTRAP_SLOT = 48947

#: 无 `__file__` 时（内存 exec / `python -c`）`_main_source()` 回落到这里，由 `_probe` 填入。
_PROBE_SOURCE: Dict[str, str] = {}

# =============================================================================
# 1. 框架 SPEC
# =============================================================================
# =============================================================================
# 1. 框架 SPEC
# =============================================================================
FRAMEWORK_SPEC: Dict[str, Any] = {
    "name": "vea_v6",
    "version": "6.0.0",
    "summary": (
        "纯标签驱动的信息流 DAG 框架：信息条目 {tags, content, image?, gen} 在图上流转，"
        "处理边 (input_tags, processor, output_tags, layer) 声明数据依赖，缺失输入由预设边"
        "从相近信息合成，结果以 tag='result' 交付；框架能把自身 SPEC 渲染成自身源码，"
        "生成的框架可逐字节重建自身。"
    ),
    "core_invariants": [
        "DAG 结构只有 {contents: [], edges: []}",
        "处理边必须有输入 tag 与输出 tag（预设边除外）",
        "dag.run() 按 layer 升序跑一轮；终止由外部检测 tag='result' 决定",
        "缺输入 tag 时预设边从相近条目合成「输出 tag = 所需输入 tag」的中间条目",
        "模型注册表 KEY → {api, key, model, params}，边内联配置优先于注册表",
        "generate_framework() 的产物逐字节等于当前框架文件",
        "manifest() 的产物逐字节等于 FRAME_MANIFEST",
    ],
    "features": [
        "ContentItem: dict 子类，键访问与属性访问并存",
        "Edge: (input_tags, processor, output_tags, layer)，processor 支持 str/dict/callable",
        "预设边：单条或多条相近信息合成，按 tag 覆盖度与相似度打分",
        "OpenAI 兼容调用：流式 SSE、注册表/内联配置、推理字段回退",
        "元构建器：把自然语言边描述编译成可运行的 V6 DAG",
        "自举：SPEC → 框架源码；源码 → 逐字节复现自身",
    ],
    "public_api": [
        "ContentItem",
        "Edge",
        "DAG",
        "DEFAULT_PRESET_EDGE",
        "SmartMockLLM",
        "OpenAICompatibleLLM",
        "load_models_file",
        "create_dag_builder_dag",
        "MetaDAGBuilder",
        "create_generation_dag",
        "bootstrap",
        "generate_framework",
        "manifest",
        "verify_self_reproduction",
    ],
}
FRAME_FILES: Tuple[str, ...] = (
    "vea_v6/parts/__init__.py",
    "vea_v6/parts/primitives.py",
    "vea_v6/parts/llm.py",
    "vea_v6/parts/dag.py",
    "vea_v6/parts/builder.py",
    MAIN_SOURCE,
    "vea_v6/__init__.py",
)

SPEC_FILES: Tuple[str, ...] = ("vea_v6/__init__.py",)

# =============================================================================
# 2. 清单与源码块（模板形态；固化后由 tools/fixpoint_v3.py 填入真实内容）
# =============================================================================
FRAME_MANIFEST = json.loads(r'''{
  "spec": {
    "name": "vea_v6",
    "version": "6.0.0",
    "summary": "纯标签驱动的信息流 DAG 框架：信息条目 {tags, content, image?, gen} 在图上流转，处理边 (input_tags, processor, output_tags, layer) 声明数据依赖，缺失输入由预设边从相近信息合成，结果以 tag='result' 交付；框架能把自身 SPEC 渲染成自身源码，生成的框架可逐字节重建自身。",
    "core_invariants": [
      "DAG 结构只有 {contents: [], edges: []}",
      "处理边必须有输入 tag 与输出 tag（预设边除外）",
      "dag.run() 按 layer 升序跑一轮；终止由外部检测 tag='result' 决定",
      "缺输入 tag 时预设边从相近条目合成「输出 tag = 所需输入 tag」的中间条目",
      "模型注册表 KEY → {api, key, model, params}，边内联配置优先于注册表",
      "generate_framework() 的产物逐字节等于当前框架文件",
      "manifest() 的产物逐字节等于 FRAME_MANIFEST"
    ],
    "features": [
      "ContentItem: dict 子类，键访问与属性访问并存",
      "Edge: (input_tags, processor, output_tags, layer)，processor 支持 str/dict/callable",
      "预设边：单条或多条相近信息合成，按 tag 覆盖度与相似度打分",
      "OpenAI 兼容调用：流式 SSE、注册表/内联配置、推理字段回退",
      "元构建器：把自然语言边描述编译成可运行的 V6 DAG",
      "自举：SPEC → 框架源码；源码 → 逐字节复现自身"
    ],
    "public_api": [
      "ContentItem",
      "Edge",
      "DAG",
      "DEFAULT_PRESET_EDGE",
      "SmartMockLLM",
      "OpenAICompatibleLLM",
      "load_models_file",
      "create_dag_builder_dag",
      "MetaDAGBuilder",
      "create_generation_dag",
      "bootstrap",
      "generate_framework",
      "manifest",
      "verify_self_reproduction"
    ]
  },
  "files": [
    {
      "path": "vea_v6/parts/__init__.py",
      "bytes": 85,
      "b64_len": 136,
      "sha256": "e0e7e626678efec1bf96840d85dba55aa31aa060185811d471d7784a72bb6bb0",
      "sig": []
    },
    {
      "path": "vea_v6/parts/primitives.py",
      "bytes": 13377,
      "b64_len": 6040,
      "sha256": "2f8cebdac71d001859531057eeb5cb5e066fde4358f50d6901d844787c9b16e6",
      "sig": [
        "def normalize_tags(tags)",
        "def content_to_str(val)",
        "def merge_payload(c1, c2)",
        "class ContentItem(dict)",
        "class Edge()"
      ]
    },
    {
      "path": "vea_v6/parts/llm.py",
      "bytes": 12722,
      "b64_len": 5572,
      "sha256": "dc813c1efb06fe0fa5d4f9623026529dff37b144450887f082180f94a1e82c1b",
      "sig": [
        "class LLMProtocol(Protocol)",
        "def extract_message_text(data)",
        "class LLMError(RuntimeError)",
        "class OpenAICompatibleLLM()",
        "class SmartMockLLM()"
      ]
    },
    {
      "path": "vea_v6/parts/dag.py",
      "bytes": 32514,
      "b64_len": 12876,
      "sha256": "894d30f72b7e58865513ec3f8d9aaabd330d7ae2d33f1b6340f2412797b5e2e9",
      "sig": [
        "def load_models_file(path)",
        "def default_text_similarity(text_a, text_b)",
        "def combine_contents(c1, c2)",
        "class DAG()"
      ]
    },
    {
      "path": "vea_v6/parts/builder.py",
      "bytes": 12180,
      "b64_len": 5400,
      "sha256": "744fc9b831de39feacab850c559e9c5e9d88914b2feb21ea3f780675dcebb538",
      "sig": [
        "def parse_json_stream(text)",
        "def parse_edges_from_description(description)",
        "def create_dag_builder_dag(similarity_threshold, ensure_result_tag)",
        "class BuilderResult()",
        "class MetaDAGBuilder()"
      ]
    },
    {
      "path": "vea_v6/parts/bootstrap.py",
      "bytes": 88742,
      "b64_len": 48947,
      "sha256": "5cd95dc7f4a1717060b6b1bf0ad19a025c216081b9f6e97743a9d59966685ed6",
      "sig": [
        "def signature(path, source)",
        "def sha_of(source)",
        "def fill_for(slot)",
        "def slot_payload(manifest_text, code_text, slot)",
        "def framework_sources()",
        "def manifest()",
        "def generate_framework(spec)",
        "def verify_self_reproduction(code)",
        "def verify_fixed_point(spec)",
        "def render_framework_code(spec)",
        "def write_framework(target_dir, files)",
        "def assemble(parts)",
        "class BootstrapResult()",
        "def create_bootstrap_dag()",
        "def normalize_spec(spec)",
        "def spec_to_text(spec)",
        "def create_generation_dag(llm)",
        "def bootstrap(description, llm)"
      ]
    },
    {
      "path": "vea_v6/__init__.py",
      "bytes": 1303,
      "b64_len": 852,
      "sha256": "0926dd2cf62d222b89472b7077097da5fc723851eb316d0019f9dc57f37015f5",
      "sig": [
        "ContentItem",
        "Edge",
        "DAG",
        "DEFAULT_PRESET_EDGE",
        "SmartMockLLM",
        "OpenAICompatibleLLM",
        "load_models_file",
        "create_dag_builder_dag",
        "MetaDAGBuilder",
        "create_generation_dag",
        "bootstrap",
        "generate_framework",
        "render_framework_code",
        "manifest",
        "verify_fixed_point",
        "verify_self_reproduction",
        "parse_edges_from_description",
        "create_bootstrap_dag",
        "FRAMEWORK_SPEC",
        "FRAME_FILES",
        "MAIN_SOURCE"
      ]
    }
  ]
}''')

FRAME_CODE = json.loads(r'''{
  "_self": "除本文件外的源码都 gzip+base64 内嵌在 code 字段；本文件的源码就是本文件自身，用 BOOTSTRAP_B64 标记占位，定长填充见 SELF_FILL，槽宽见 BOOTSTRAP_SLOT",
  "files": [
    {
      "path": "vea_v6/parts/__init__.py",
      "key": "VEA_V6_PARTS___INIT___PY_B64",
      "code": "H4sIAAAAAAAC/1NSUgpzdVQIM1N42tb6snnF03XznvdteNrR9ryz/f2ejoKizNzMksyy1GIFfYWcnFwgmZKYDiSTSjNzUlKLQKz8/JLikqLEgvd7Oh83NCkpKXEBAHFQPr1VAAAA"
    },
    {
      "path": "vea_v6/parts/primitives.py",
      "key": "VEA_V6_PARTS_PRIMITIVES_PY_B64",
      "code": "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"
    },
    {
      "path": "vea_v6/parts/llm.py",
      "key": "VEA_V6_PARTS_LLM_PY_B64",
      "code": "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"
    },
    {
      "path": "vea_v6/parts/dag.py",
      "key": "VEA_V6_PARTS_DAG_PY_B64",
      "code": "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"
    },
    {
      "path": "vea_v6/parts/builder.py",
      "key": "VEA_V6_PARTS_BUILDER_PY_B64",
      "code": "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"
    },
    {
      "path": "vea_v6/__init__.py",
      "key": "VEA_V6___INIT___PY_B64",
      "code": "H4sIAAAAAAAC/31Uz08TQRS+718x2ZMma+XEgcRDLVvS2FrSAh6MmS7d12bC7s5mdlrhBkagTYCSKCY0Rg9KjAbBgzExVP8ZmVZP/AvOdpl1Szfu6c33zXvf+zWr6/qKmUUrs0hsn4rBptg+Eb13V4NOddHMoeGr3cuLb1eD7tWgL3qfxFYfMbgD6z5l/NfmM00bvXw77ByOuh/F+ffh69Po/qj/XBxcjI6Opctwr4tqGFuOg3ENjT7/FIf7ovNenPWlxu8PW6jmW4wHd1cp5QFnlp/xN+bmAtL0LN5iUBtr72ni7M3o4Mvljxeid355cTI8+irtP7v7Yuc4zEPXdU3DuA0sINTDGN1D+mxmJjMzhsfiEnqsIfnpOepx8HiBg6sbEWTaTVD2fHYhNs18drm4hBcrZtVcwub8gqmoqivTLtH6WrFYUljZBy9byFHXtzhZdSBBOdSysUttcALcIE4sVmdgccC21cSrLeLYwEJbsSXglkznfsTc8GmCB0wKyXoTLnEfFXB9DXCDWS48pWxNMQy8UC/GcV3mp0jX8kgDAq7OsrWksSFzXwcb+5R4N5kAnAZm4DNqt+phVoqX8w0Ag+ywLJ1RF9sQ1Bnxk3euS4qTT1aUr2RL5qNy5QEOV3ICxflC0azG3coWHuJqebmSC6f0RNNCNdQGC7dnM+Mty8QCiLjhDqNbY9dEMOMfEGtGWCJ8BMTBomNaDRPM5MAiano6Ea7aH51SJxVR03OZwKemYmi30/oSbZjqyuTeGSh9Sw30v9GmqEgfpZDysgwkJQ1086WkxHEcV8VJeXIGSr7NFHefEZdw0oZARUn8EQwU/gu0v7ygSvsXBQAA"
    },
    {
      "path": "vea_v6/parts/bootstrap.py",
      "key": "BOOTSTRAP_B64",
      "code": "BOOTSTRAP_B64"
    }
  ]
}''')


# =============================================================================
# 3. 工具
# =============================================================================
def _decode(text: str) -> bytes:
    return gzip.decompress(base64.b64decode(text.encode("ascii")))


def signature(path: str, source: str) -> List[str]:
    """提取公开签名：顶层 `def`/`class` 声明（跳过私有名，含参数名）。"""
    if path in SPEC_FILES:
        m = re.search(r"^__all__\s*=\s*\[(.*?)\]", source, re.S | re.M)
        if not m:
            raise ValueError(f"{path}: SPEC 文件必须有 __all__")
        return re.findall(r'"([^"]+)"', m.group(1))
    tree = ast.parse(source)
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                out.append(f"def {node.name}({', '.join(a.arg for a in node.args.args)})")
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                out.append(f"class {node.name}({', '.join(ast.unparse(b) for b in node.bases)})")
    return out


def _gz(raw: bytes) -> bytes:
    return gzip.compress(raw, 9, mtime=0)


def _gzlen(src: str) -> int:
    """源码的 gzip 体积（mtime 固定，结果确定）。"""
    return len(_gz(src.encode("utf-8")))


def _b64of(raw: bytes) -> str:
    return base64.b64encode(_gz(raw)).decode("ascii")


_MAN_RE = re.compile(r"^FRAME_MANIFEST = json\.loads\(r'''.*?'''\)", re.S | re.M)
_CODE_RE = re.compile(r"^FRAME_CODE = json\.loads\(r'''.*?'''\)", re.S | re.M)
_SLOT_RE = re.compile(r"^BOOTSTRAP_SLOT = \d+", re.M)


def _render(source: str, manifest_text: str, code_text: str) -> str:
    """把清单与代码块写进主源；不动 SELF_FILL、不动 BOOTSTRAP_SLOT。"""
    out, n1 = _MAN_RE.subn(lambda _m: "FRAME_MANIFEST = json.loads(r'''" + manifest_text + "''')", source, count=1)
    if n1 != 1:
        raise ValueError("主源缺少 FRAME_MANIFEST 块")
    out, n2 = _CODE_RE.subn(lambda _m: "FRAME_CODE = json.loads(r'''" + code_text + "''')", out, count=1)
    if n2 != 1:
        raise ValueError("主源缺少 FRAME_CODE 块")
    return out


def _set_slot(source: str, pad: int) -> str:
    """写 BOOTSTRAP_SLOT 的**值**（只改数字，不改字节数之外的任何东西）。"""
    out, n = _SLOT_RE.subn(f"BOOTSTRAP_SLOT = {pad}", source, count=1)
    if n != 1:
        raise ValueError("主源缺少 BOOTSTRAP_SLOT 行")
    return out


def _code_text(payloads: Dict[str, str]) -> str:
    """FRAME_CODE 的规范渲染：其它文件带 payload，主源只放自指标记。"""
    files = [{"path": p, "key": _KEY_OF(p), "code": payloads[p]} for p in payloads if p != MAIN_SOURCE]
    files.append({"path": MAIN_SOURCE, "key": SELF_MARK, "code": SELF_MARK})
    return json.dumps(
        {
            "_self": (
                "除本文件外的源码都 gzip+base64 内嵌在 code 字段；"
                "本文件的源码就是本文件自身，用 " + SELF_MARK + " 标记占位，"
                "定长填充见 SELF_FILL，槽宽见 BOOTSTRAP_SLOT"
            ),
            "files": files,
        },
        ensure_ascii=False,
        indent=2,
    )


def _KEY_OF(path: str) -> str:
    return path.replace("/", "_").replace(".", "_").upper() + "_B64"


def _is_prime(n: int) -> bool:
    """素数判定：槽宽取素数，避免"填充长度"与"清单数字"形成短周期共振。"""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _next_prime(n: int) -> int:
    k = max(2, n)
    while not _is_prime(k):
        k += 1
    return k


#: 主源在清单里的 sha 位置要填的占位（等长于 sha256 十六进制串）
_SHA_HOLE = "0" * 64


def _skeleton_sha(source: str) -> str:
    """主源"骨架"的 sha256：清单里**主源自己那一行的 `sha256` 值**先涂成占位，再取整文哈希。

    两个坑都要避开：
    - 若直接记 `sha(full_source)`，写进清单就改了全文 ⇒ 永远对不上；
    - 若只涂掉 `SELF_FILL` 行，那清单里的这个值仍然参与哈希 ⇒ 仍是自指。

    涂掉"这个值自己"之后，等式变成 `h = sha(f(x))`，`x` 不再影响 `f(x)`，一遍就能收敛：
    输出 `h` 的行与涂掉 `h` 的行是同一行、长度也相同（占位是等长 64 个 `0`），所以**不改变字节数**。
    """
    hole = source
    if BOOTSTRAP_SLOT:
        hole = _fill_self_sha_placeholder(hole)
    return sha_of(hole)


def _fill_self_sha_placeholder(source: str) -> str:
    """把清单里主源那一行的 sha256 值替换成等长占位（只改这一处，字节数不变）。"""
    pat = re.compile(
        r'("path":\s*"' + re.escape(MAIN_SOURCE) + r'",\s*"bytes":\s*\d+,\s*"b64_len":\s*\d+,\s*"sha256":\s*")[0-9a-f]{64}(")'
    )
    out, n = pat.subn(lambda m: m.group(1) + _SHA_HOLE + m.group(2), source, count=1)
    return out if n == 1 else source


def sha_of(source: str) -> str:
    """源码自身的 sha256（十六进制）。"""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def fill_for(slot: int) -> str:
    """槽内该填的字符串：`(slot - len(sha256))` 个 'x'。

    `sha256` 是固定 64 位十六进制，所以填充长度只由 slot 决定——**内容本身不含源码**，
    真正的自指信息在"能算出 sha 并通过校验"这件事上。
    """
    return "x" * max(0, int(slot) - 64)


def slot_payload(manifest_text: str, code_text: str, slot: int) -> str:
    """给定槽宽，算出这个槽里应当放的 base64 文本（长度恰为 slot）。"""
    base = _set_slot(_render(_main_source(), manifest_text, code_text), slot)
    return fill_for(slot) + sha_of(base)


def _solve(
    main_src: str,
    manifest_of,
    code_text: str,
    self_bytes: int = 0,
    start_pad: int = 0,
) -> Tuple[int, str]:
    """求槽宽 pad 与最终主源（含把清单里主源那一行的 sha256 收敛到位）。

    两层解耦仍然是基础：`SELF_FILL` 是定长占位符、`BOOTSTRAP_SLOT = pad` 只写数字，
    所以"填充"不是解不动点的难点；难点只剩清单里主源 `bytes`（= 渲染结果自身长度）
    与 `sha256`（= 涂掉自己那一格之后的整文哈希）两个自指量。两者都已变成**一遍可算**：
    - `bytes`：用上一轮的测量值代入（外层 `assemble` 迭代到相等）；
    - `sha256`：涂掉自己那一格 ⇒ `x` 不影响 `f(x)`，代入即收敛。

    `start_pad`：候选起点。**传磁盘上已固化的槽宽**时，若它本身就闭合，就原样返回它——
    这是"求解结果 == 磁盘主源"能成立的前提。不传则按 `z + 4096` 起算（首次固化用）。
    """
    if start_pad:
        cand = _set_slot(_render(main_src, manifest_of(start_pad, "", self_bytes), code_text), start_pad)
        if _gzlen(cand) <= start_pad and _reproduces(cand):
            return int(start_pad), cand
    body = _render(main_src, manifest_of(32127, "", self_bytes), code_text)
    z = _gzlen(_set_slot(body, 32127))
    # 余量：渲染与最终形态之间的常数差（缩进/位数抖动），几千字节足够且不影响正确性。
    #
    # 🔴 `pad` 是**多解**的：只要 `zz <= pad` 内层判据就通过（槽里多余的 'F' 会被 gzip
    # 压掉），实测 46447 / 46451 / 46453 / 46500 / 47000 全部 `_reproduces=True`。
    # 所以"重新求解"和"磁盘上已固化的"能给出两个都自洽但不同的 pad。
    #
    # ⚠️ 本模块内的求解器**必须与 `tools/fixpoint_v3.py` 的 `solve()` 逐字一致**：
    # `verify_fixed_point()` 是做"模块解得的主源 == 磁盘主源"的相等断言，两边只要
    # 差一个常数余量（以前是 `z+4096` vs `z`）就会永久 FAIL。曾经把两边一起改成
    # "最小素数"试图收紧判据，结果模块内因为多处注释变长、z 涨了 1.4k 字节，
    # 外加共享全局函数里有状态，一次没收敛。结论：**先收紧的是判据的确定性，
    # 不是 pad 的大小**——同一个 z 一旦算出来，两条路径必须走到同一个起点。
    pad = _next_prime(max(z + 4096, 256))
    for _ in range(64):
        src = _set_slot(_render(main_src, manifest_of(pad, "", self_bytes), code_text), pad)
        zz = _gzlen(src)
        if zz > pad:
            pad = _next_prime(max(pad + 32, zz + 64))
            continue
        cur = ""
        for _ in range(12):
            sha = _skeleton_sha(src)  # 涂掉主源自己那一格后取哈希 ⇒ 与 .sha 取值无关
            if sha == cur:
                if _reproduces(src):
                    return pad, src
                break
            cur, src = sha, _set_slot(_render(main_src, manifest_of(pad, sha, self_bytes), code_text), pad)
            if _gzlen(src) > pad:
                break
        else:
            raise RuntimeError("清单 sha 回填未收敛")
        pad = _next_prime(max(pad + 32, zz + 64))
    raise RuntimeError(f"固定点未收敛（最后试到 pad={pad}）")


def _reproduces(src: str) -> bool:
    """内层判据：把 src 当主源跑一遍，`generate_framework()` 是否逐字节回放出 src 自己。

    这里**只验 payload 闭合**，不验清单里 `bytes` 是否等于真实长度。原因是不动点分两层：
    内层解 `pad` + 骨架 `sha`（此时 `bytes` 还是占位），外层 `assemble` 迭代 `bytes` 到真值，
    最后由 `verify_self_reproduction()` 做完整闸门。混在一起会让两层互相锁死（实测不收敛）。
    """
    try:
        return _probe(src)["generate_framework"]()[MAIN_SOURCE] == src
    except Exception:
        return False


def _probe(src: str) -> Dict[str, Any]:
    """在独立模块对象里执行源码（dataclass 要求模块已注册进 sys.modules）。"""
    import sys
    import types

    name = "_vea_probe"
    # dataclass / typing 要求模块在 sys.modules 里、且拿到的是**模块对象本身**；
    # 只 `sys.modules[name] = mod` 而之前有同名残留为 None 时，会报
    # `'NoneType' object has no attribute '__dict__'`。
    mod = types.ModuleType(name)
    mod.__dict__["__name__"] = name
    sys.modules[name] = mod
    exec(compile(src, "<probe>", "exec"), mod.__dict__)  # noqa: S102 - 只执行自家源码
    mod.__dict__["_PROBE_SOURCE"]["current"] = src  # 无 __file__ 时 _main_source() 的回落值
    return mod.__dict__


# =============================================================================
# 4. 只读视图
# =============================================================================
def _main_source() -> str:
    """本文件在磁盘上的内容；没有 `__file__`（内存里 exec）时退化为「重建结果就是本段源码」。"""
    path = globals().get("__file__")
    if path:
        from pathlib import Path

        return Path(path).read_text(encoding="utf-8")
    return _PROBE_SOURCE.get("current", "")


def _outer_sources() -> Dict[str, str]:
    """除主源外的全部源码（主源在 FRAME_CODE 里只是自指标记，不能解码）。"""
    return {
        item["path"]: _decode(item["code"]).decode("utf-8")
        for item in FRAME_CODE.get("files", [])
        if item["path"] != MAIN_SOURCE
    }


def _contract_violation(path: str, code: str) -> str:
    """模型产出与原文件比对：顶层契约缺失即判不合格，返回原因（空串=通过）。

    判据只看**顶层骨架**，不看函数体——
    - 语法必须能过；
    - 原文件顶层有的 `class` / `def` / 赋值名，新文件一个都不许少（允许新增）；
    - 文件体积不得坍缩到原文件的 40% 以下（掏空型改写的主要外形特征）；
    - SPEC 文件必须仍带 `__all__`。

    为什么不能用"体积"单条判：模型确实可能写出更紧凑的等价实现，
    所以主判据是**符号集合**，体积只作兜底（防"结构齐全但内容注水/清空"）。
    """
    orig = _outer_sources().get(path)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"语法错误: {exc.msg} (line {exc.lineno})"
    if path in SPEC_FILES:
        has_all = any(
            isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in n.targets)
            for n in tree.body
        )
        if not has_all:
            return "SPEC 文件缺 __all__"
    if not orig:
        return ""
    try:
        old_tree = ast.parse(orig)
    except SyntaxError:
        return ""

    def _tops(t: ast.AST) -> dict:
        """顶层定义名 -> (节点, 源码行数)。只要名字，用 `_tops(...).keys()`。"""
        out: dict = {}
        src = t.body  # type: ignore[attr-defined]
        for n in src:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                span = (n.end_lineno or n.lineno) - n.lineno + 1
                out[n.name] = span
            elif isinstance(n, ast.Assign):
                for tgt in n.targets:
                    if isinstance(tgt, ast.Name):
                        out[tgt.id] = (n.end_lineno or n.lineno) - n.lineno + 1
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                out[n.target.id] = (n.end_lineno or n.lineno) - n.lineno + 1
        return out

    old, new = _tops(old_tree), _tops(tree)
    lost = sorted(set(old) - set(new))
    if lost:
        return f"顶层定义缺失 {len(lost)} 个（{', '.join(lost[:5])}{'…' if len(lost) > 5 else ''}）"
    # 名字齐全但函数体被大幅削掉，同样是掏空。实测 `primitives.py` 的 `Edge`
    # 从 174 行掉到 130 行（-25%），而整文件只掉 15%——**总体体积判据抓不住**，
    # 必须按定义逐个比。阈值取 50%：允许重写变紧凑，不允许砍掉大半逻辑。
    shrunk = [
        f"{k}({old[k]}→{new[k]}行)"
        for k in old
        if old[k] >= 20 and new[k] < old[k] // 2
    ]
    if shrunk:
        return f"定义被削掉过半 {len(shrunk)} 个（{', '.join(shrunk[:4])}）"
    if len(code) < len(orig) * 2 // 5:
        return f"体积坍缩到 {len(code)}B（原 {len(orig)}B），疑似截断"
    return ""


def framework_sources() -> Dict[str, str]:
    """框架当前源码 `{路径: 文本}`（来自 FRAME_CODE + 磁盘上的主源）。"""
    files = _outer_sources()
    files[MAIN_SOURCE] = _main_source()
    return {p: files[p] for p in FRAME_FILES}


def manifest() -> str:
    """渲染框架清单（规范 JSON）；对同一份源码恒定。

    主源那一行的语义：
    - `bytes` = 主源真实字节数（也就是"整份渲染结果的长度"，自指但只差固定常数 ⇒ 迭代两轮即定）；
    - `b64_len` = 槽宽（`BOOTSTRAP_SLOT`，见下方注释）；
    - `sha256` = **骨架 sha**（涂掉这一格自己之后的整文哈希，见 `_skeleton_sha`）。
    """
    files = framework_sources()
    docs = []
    for p in FRAME_FILES:
        raw = files[p].encode("utf-8")
        docs.append(
            {
                "path": p,
                "bytes": len(raw),
                # 主源的 b64_len 就是槽宽（冻结在 BOOTSTRAP_SLOT）；它不等于自身 payload 的长度，
                # 因为槽里放的是"占位符"而不是内容——把两者混淆会让自检恒失败。
                "b64_len": BOOTSTRAP_SLOT if p == MAIN_SOURCE else 4 * ((_gzlen(files[p]) + 2) // 3),
                # 主源记的是"去掉填充行后的骨架"的 sha256：填充行长度是槽宽派生的，
                # 若直接记全文 sha，则写进清单就改掉全文、永远对不上（二次自指陷阱）。
                # 骨架固定不变，所以这一行是**可验证且稳定**的：任何人都能复算。
                "sha256": _skeleton_sha(files[p]) if p == MAIN_SOURCE else sha_of(files[p]),
                "sig": signature(p, files[p]),
            }
        )
    return json.dumps({"spec": FRAMEWORK_SPEC, "files": docs}, ensure_ascii=False, indent=2)


def _manifest_of_spec(spec_obj: Dict[str, Any], files: Optional[Dict[str, str]] = None) -> str:
    """按给定 SPEC 生成清单文本（与 `manifest()` 同构，但 SPEC 可指定）。"""
    src_map = files or framework_sources()
    docs = []
    for p in FRAME_FILES:
        raw = src_map[p].encode("utf-8")
        docs.append(
            {
                "path": p,
                "bytes": len(raw),
                "b64_len": BOOTSTRAP_SLOT if p == MAIN_SOURCE else 4 * ((_gzlen(src_map[p]) + 2) // 3),
                "sha256": _skeleton_sha(src_map[p]) if p == MAIN_SOURCE else sha_of(src_map[p]),
                "sig": signature(p, src_map[p]),
            }
        )
    return json.dumps({"spec": normalize_spec(spec_obj), "files": docs}, ensure_ascii=False, indent=2)


def generate_framework(spec: Optional[Dict[str, Any]] = None, *, slot: Optional[int] = None) -> Dict[str, str]:
    """重建整个框架源码 `{路径: 文本}`，逐字节等于被内嵌的那份。

    主源的重建 = 把清单/代码块换成真值 ⇒ 把 `BOOTSTRAP_SLOT` 写回冻结的槽宽。
    `SELF_FILL` **原样保留**（它是定长占位符，不是内容槽）——
    因为它本身就是"定长"所以代表不了源码，能代表源码的是**槽宽**与**清单里的 sha256**。
    """
    out = _outer_sources()
    width = BOOTSTRAP_SLOT if slot is None else int(slot)
    man = json.dumps(FRAME_MANIFEST, ensure_ascii=False, indent=2)
    code = json.dumps(FRAME_CODE, ensure_ascii=False, indent=2)
    out[MAIN_SOURCE] = _set_slot(_render(_main_source(), man, code), width)
    return out


def verify_self_reproduction(code: Optional[str] = None) -> Tuple[bool, str]:
    """自复现判定。传 `code` 时就地校验那段源码（把它当主源跑一遍），不传则校验磁盘版本。"""
    if code is not None:
        try:
            ns = _probe(code)
            return bool(ns["verify_self_reproduction"]()[0]), str(ns["verify_self_reproduction"]()[1])
        except Exception as exc:  # noqa: BLE001 - 校验失败一律视为不闭合
            return False, f"执行失败: {exc}"
    return _verify_disk()
    """自复现判定（三级）：

    1. **文件级**：`generate_framework()` 重建出的每个文件是否逐字节等于当前文件；
    2. **清单级**：清单里记录的每个文件 sha256 是否等于该文件真实 sha256
       （**不要求**"清单自己的 sha 等于清单里写的那一行"——那是二次自指，必须迭代到不动点，
       而 sha 回填序列实测不收敛，追它只会得到恒 FAIL 的死判据）；
    3. **固定点级**：重建出来的主源再生成一次，是否还是它自己（生成器闭合）。
    """
    return _verify_disk()


def _verify_disk() -> Tuple[bool, str]:
    cur = framework_sources()
    regen = generate_framework()
    diffs = [p for p in FRAME_FILES if cur.get(p) != regen.get(p)]
    if diffs:
        return False, f"源码不一致: {diffs}"

    recorded = {d["path"]: d.get("sha256") for d in FRAME_MANIFEST.get("files", [])}
    wrong = [
        p
        for p in FRAME_FILES
        if recorded.get(p) != (_skeleton_sha(cur[p]) if p == MAIN_SOURCE else sha_of(cur[p]))
    ]
    if wrong:
        return False, f"清单 sha 与真实源码不符: {wrong}"

    # `bytes` 也必须核：它是"渲染结果自身长度"这个自指量，如果只写不验，
    # 求解器就有自由变量，同一份源码能解出多个 `pad`（实测 43411 与 43427 都"自洽"）。
    sizes = {d["path"]: d.get("bytes") for d in FRAME_MANIFEST.get("files", [])}
    bad_sizes = [p for p in FRAME_FILES if sizes.get(p) != len(cur[p].encode("utf-8"))]
    if bad_sizes:
        return False, f"清单 bytes 与真实长度不符: {bad_sizes}"

    gen_again = _probe(regen[MAIN_SOURCE])["generate_framework"]()[MAIN_SOURCE]
    if gen_again != regen[MAIN_SOURCE]:
        return False, "二代生成与一代不一致（生成器未闭合）"

    total = sum(len(v.encode()) for v in regen.values())
    return True, f"{len(FRAME_FILES)} 个文件 / {total} 字节逐字节一致；清单 sha 全部相符；二代复现一致"


def verify_fixed_point(spec: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """同一份源码解一次固定点是否幂等（`assemble` 两次结果逐字节相同）。

    与 `verify_self_reproduction` 的区别：这里验的是"求解器本身无漂移"，
    那边验的是"生成器能重建自己"。两者都过才叫真闭合。
    """
    try:
        a = assemble(framework_sources(), spec=normalize_spec(spec))
        b = assemble(framework_sources(), spec=normalize_spec(spec))
    except Exception as exc:  # noqa: BLE001 - 求解失败就是不稳定
        return False, f"求解失败: {exc}"
    diff = [p for p in FRAME_FILES if a.get(p) != b.get(p)]
    if diff:
        return False, f"两次求解不一致: {diff}"
    if a.get(MAIN_SOURCE) != framework_sources().get(MAIN_SOURCE):
        return False, "求解结果与磁盘主源不一致（需重新固化：tools/fixpoint_v3.py）"
    return True, f"固定点幂等：{len(FRAME_FILES)} 个文件 / {sum(len(v.encode()) for v in a.values())} 字节"


def render_framework_code(spec: Optional[Dict[str, Any]] = None) -> str:
    """按 SPEC 渲染出完整的框架主源文本（含清单与全部内嵌源码）。"""
    files = assemble(framework_sources(), spec=normalize_spec(spec))
    return files[MAIN_SOURCE]


def write_framework(target_dir: str, files: Optional[Dict[str, str]] = None) -> List[str]:
    from pathlib import Path

    root = Path(target_dir)
    data = files if files is not None else generate_framework()
    written = []
    for path, text in data.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        written.append(str(p))
    return written


# =============================================================================
# 5. assemble：把任意文件集变成自洽框架（模型侧唯一写入口）
# =============================================================================
def _self_bytes_on_disk() -> int:
    """磁盘主源清单里记的 `bytes`（主源自身真实字节数）；拿不到就回落到 0（=还没算过）。

    `assemble` 的迭代起点必须是"主源当前状态"，不能固定成某个常数：
    传 100000 会让首轮解出的 pad 与磁盘固化的 pad 不同（实测 48437 vs 48407），
    于是 `verify_fixed_point()` 的相等断言恒 FAIL，而自复现却一直 OK。
    """
    m = re.search(r'"path":\s*"' + re.escape(MAIN_SOURCE) + r'",\s*"bytes":\s*(\d+)', _main_source())
    return int(m.group(1)) if m else 0


def _self_sha_on_disk() -> str:
    """磁盘主源清单里记的骨架 `sha256`；拿不到返回空串（表示待重算）。

    与 `_self_bytes_on_disk` 同理：主源那一格是**自描述元数据**，`assemble` 只负责
    把全部其它文件与 `pad` 收敛到固定点，不该顺手把这一格重算成另一个值。
    """
    m = re.search(r'"path":\s*"' + re.escape(MAIN_SOURCE) + r'",\s*"bytes":\s*\d+,\s*"b64_len":\s*\d+,\s*"sha256":\s*"([0-9a-f]{64})"', _main_source())
    return m.group(1) if m else ""


def _slot_on_disk() -> int:
    """磁盘主源清单里记的 `b64_len`（= 槽宽 pad）；拿不到返回 0。"""
    m = re.search(r'"path":\s*"' + re.escape(MAIN_SOURCE) + r'",\s*"bytes":\s*\d+,\s*"b64_len":\s*(\d+)', _main_source())
    return int(m.group(1)) if m else 0


def assemble(parts: Dict[str, str], *, spec: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """补清单、解固定点、自校验，返回可逐字节复现的文件集。"""
    files = dict(parts)
    missing = [p for p in FRAME_FILES if p not in files]
    if missing:
        raise ValueError(f"assemble 缺少文件: {missing}")
    for p in FRAME_FILES:
        try:
            ast.parse(files[p])
        except SyntaxError as exc:
            raise ValueError(f"{p} 语法错误: {exc}") from exc

    spec_obj = spec or FRAMEWORK_SPEC
    others = [p for p in FRAME_FILES if p != MAIN_SOURCE]
    raw = {p: files[p].encode("utf-8") for p in others}
    payloads = {p: _b64of(raw[p]) for p in others}
    code_text = _code_text(payloads)

    def manifest_text(pad: int, self_bytes: int, self_sha: str) -> str:
        docs = []
        for p in FRAME_FILES:
            if p == MAIN_SOURCE:
                # bytes 是真实字节数（与 pad 无关）；b64_len 就是槽宽（与 pad 同义）。
                # `sha256` 由外层 `assemble` 迭代到收敛后固定传入（见下方注释）。
                docs.append(
                    {
                        "path": p,
                        "bytes": self_bytes or len(str(pad)),
                        "b64_len": pad,
                        "sha256": self_sha,
                        "sig": signature(p, files[p]),
                    }
                )
            else:
                docs.append(
                    {
                        "path": p,
                        "bytes": len(raw[p]),
                        "b64_len": 4 * ((len(_gz(raw[p])) + 2) // 3),
                        "sha256": hashlib.sha256(raw[p]).hexdigest(),
                        "sig": signature(p, files[p]),
                    }
                )
        return json.dumps({"spec": spec_obj, "files": docs}, ensure_ascii=False, indent=2)

    # 迭代起点必须是**主源当前状态**（磁盘已固化过就取清单里那个 bytes），
    # 并用磁盘上已有的 `b64_len`（=槽宽）起步。若从常数 100000 起步，`_solve` 的
    # 起点 `z + 4096` 里 z 随输入变，差一轮就解出不同的 pad（48397 / 48407 / 48661
    # 都出现过），于是"求解结果 == 磁盘主源"的相等断言恒 FAIL，而自复现却一直 OK。
    #
    # 主源清单那一行的 `sha256` 是**骨架哈希**（会把这一格自己涂成 0 再整文哈希），
    # 所以它是"由候选文本唯一决定"的量：必须**先收敛再固定**，不能直接抄磁盘上的旧值
    # （换了 SPEC 之后旧值必然对不上，实测报"清单 sha 与真实源码不符"）。
    self_bytes = _self_bytes_on_disk()
    disk_pad = _slot_on_disk()
    src = files[MAIN_SOURCE]
    for _ in range(8):
        pad, sha = "", ""
        for _ in range(12):
            pad2, cand = _solve(
                files[MAIN_SOURCE],
                lambda pad_, sh, nbytes: manifest_text(pad_, nbytes, sh),
                code_text,
                self_bytes,
                start_pad=disk_pad,
            )
            sha2 = _skeleton_sha(cand)
            if sha2 == sha and pad2 == pad:
                pad, src = pad2, cand
                break
            pad, sha = pad2, sha2
            disk_pad = 0  # 只有第一轮用磁盘槽宽起步，之后按求解值走
        else:
            raise RuntimeError("assemble: 骨架 sha 未收敛")
        real = len(src.encode("utf-8"))
        if real == self_bytes:
            break
        self_bytes = real
    else:
        raise RuntimeError(f"assemble: 字节数未收敛（停在 {self_bytes}）")

    ns = _probe(src)
    gen = ns["generate_framework"]()
    want = {p: (files[p] if p != MAIN_SOURCE else src) for p in FRAME_FILES}
    bad = [p for p in FRAME_FILES if gen.get(p) != want[p]]
    if bad:
        raise RuntimeError(f"assemble: 复现失败 {bad}")
    out = {p: files[p] for p in FRAME_FILES if p != MAIN_SOURCE}
    out[MAIN_SOURCE] = src
    return out

# =============================================================================
# 6. 生成 DAG
# =============================================================================
@dataclass
class BootstrapResult:
    """一次生成的结果。"""

    files: Dict[str, str]
    manifest_text: str
    okay: bool
    reason: str = ""
    context: List[ContentItem] = field(default_factory=list)

    @property
    def stable(self) -> bool:
        """兼容旧字段名：`okay` 的同义词。"""
        return self.okay

    @property
    def framework_code(self) -> str:
        """主源全文（即"这份框架源码"）。"""
        return self.files.get(MAIN_SOURCE, "")

    @property
    def spec(self) -> Dict[str, Any]:
        """本次生成用的 SPEC（从结果里的清单回读，保证与源码内嵌的一致）。"""
        try:
            data = json.loads(self.manifest_text) if self.manifest_text.strip().startswith("{") else {}
        except (json.JSONDecodeError, TypeError):
            data = {}
        if isinstance(data, dict) and isinstance(data.get("spec"), dict):
            return data["spec"]
        for item in self.context:
            try:
                obj = json.loads(item["content"])
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            if isinstance(obj, dict) and isinstance(obj.get("spec"), dict):
                return obj["spec"]
        return {}

    def write_to(self, target_dir: str) -> List[str]:
        return write_framework(target_dir, self.files)


def create_bootstrap_dag(*args: Any, **kwargs: Any) -> DAG:
    """兼容旧名：生成框架的 DAG（等价 `create_generation_dag`）。"""
    return create_generation_dag(*args, **kwargs)


def normalize_spec(spec: Union[Dict[str, Any], str, None]) -> Dict[str, Any]:
    """SPEC 规范化：键序固定、列表去重、缺键回落默认，保证渲染确定。"""
    if spec is None:
        base: Dict[str, Any] = dict(FRAMEWORK_SPEC)
    elif isinstance(spec, str):
        text = spec.strip()
        base = dict(FRAMEWORK_SPEC)
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    base = dict(parsed)
            except json.JSONDecodeError:
                base["summary"] = text
    elif isinstance(spec, dict):
        base = dict(spec)
    else:
        raise TypeError(f"SPEC 必须是 dict/JSON/None，实际 {type(spec).__name__}")

    out: Dict[str, Any] = {
        "name": str(base.get("name") or FRAMEWORK_SPEC["name"]),
        "version": str(base.get("version") or FRAMEWORK_SPEC["version"]),
        "summary": str(base.get("summary") or FRAMEWORK_SPEC["summary"]).strip(),
    }
    for key in ("core_invariants", "features", "public_api"):
        raw_val = base.get(key, FRAMEWORK_SPEC[key])
        items = raw_val.splitlines() if isinstance(raw_val, str) else list(raw_val)
        out[key] = list(dict.fromkeys(str(x).strip() for x in items if str(x).strip()))
    return out


def spec_to_text(spec: Union[Dict[str, Any], str, None] = None) -> str:
    return json.dumps(normalize_spec(spec), ensure_ascii=False, indent=2)


#: 三段式提示词（生成 DAG 的每条边问一次模型）
PROMPTS: Dict[str, str] = {
    "spec": (
        "你在为下述框架写可执行规格（SPEC）。只输出 JSON，键为 name/version/summary/"
        "core_invariants/features/public_api，不要解释、不要 markdown 围栏。\n"
        "框架现状：\n{current}\n\n用户补充描述：\n{input}"
    ),
    "skeleton": (
        "框架文件与公开签名如下（JSON）。请输出你打算保留的签名，结构 "
        '{{"files":[{{"path":...,"sig":[...]}}]}}。\n'
        "硬约束：{spec_file} 的签名原样保留；public_api 中的每个名字都必须出现在某个文件的签名里。\n"
        "{sig_list}"
    ),
    "flesh": (
        "重写下面这个文件（只输出代码本体，不要 markdown 围栏）。\n"
        "约束：只用标准库与 `from vea_v6.parts…` 兄弟模块；import 期不做 IO；保留原签名。\n"
        "文件：{path}\n原源码：\n{source}"
    ),
}


def create_generation_dag(
    llm: Optional[Any] = None,
    *,
    models: Optional[Dict[str, Dict[str, Any]]] = None,
    model_key: str = "laptop-6.8",
    rewrite_files: Sequence[str] = (),
) -> DAG:
    """创建框架生成 DAG（标准 V6 DAG；三条 layer-1 边汇聚成 skeleton_code）。

    没显式传 `llm` 时**必须**造一个真的 `OpenAICompatibleLLM`。
    不能交给 `DAG()` 的默认值——那里会静默退化成 `SmartMockLLM`，
    整条链能在 1 秒内"跑通"却一次 HTTP 请求都没发过（假通过）。
    """
    if llm is None:
        llm = OpenAICompatibleLLM()
    if models is None:
        models = load_models_file()
    dag = DAG("V6_Framework_Generation_DAG", preset_edge=DEFAULT_PRESET_EDGE, llm=llm, models=models)
    base = dict(framework_sources())
    # 默认让模型重写**全部非主源文件**（主源自带自指槽，只能由固定点求解器生成）。
    # 若默认留空，模型一次文件都不写，整条链只是把磁盘源码原样回放，
    # 那种"1 秒跑通"是假通过——没验证模型任何能力。
    targets = [p for p in (rewrite_files or [p for p in FRAME_FILES if p != MAIN_SOURCE]) if p != MAIN_SOURCE]


    def _ask(prompt: str) -> str:
        return dag.ask_sync(prompt, model_key)

    def _edge_spec(content: str) -> str:
        current = spec_to_text(FRAMEWORK_SPEC)
        if not content.strip():
            return current
        return spec_to_text(_extract_code(_ask(PROMPTS["spec"].format(current=current[:4000], input=content)), "json"))

    def _edge_skeleton(spec_text: str) -> str:
        spec = normalize_spec(spec_text)
        sig_list = json.dumps({p: signature(p, base[p]) for p in FRAME_FILES}, ensure_ascii=False, indent=1)
        raw = _ask(PROMPTS["skeleton"].format(spec_file=SPEC_FILES[0], sig_list=sig_list))
        return json.dumps({"spec": spec, "plan": _merge_plan(spec, raw)}, ensure_ascii=False)

    def _edge_flesh(spec_text: str) -> str:
        spec = normalize_spec(spec_text)
        jobs = [{"path": p, "prompt": PROMPTS["flesh"].format(path=p, source=_clip(base[p]))} for p in targets]
        return json.dumps({"spec": spec, "jobs": jobs}, ensure_ascii=False)

    def _edge_skeleton_code(merged: str) -> str:
        """把预设边拼起来的三段 JSON 还原成 (spec, plan-with-flesh)。

        三段来源不同、key 会撞车，不能简单 `data.update()`：
          - spec 边     -> 裸 SPEC 对象（name/summary/...）
          - skeleton 边 -> {"spec", "plan"}
          - flesh 边    -> {"spec", "jobs"}
        `jobs` 必须并入 `plan["flesh"]`，否则 `_edge_frame_code` 拿不到重写任务，
        模型一次文件都不会写（曾经的"假通过"就是这么来的）。
        """
        spec_obj: Optional[Dict[str, Any]] = None
        plan: Dict[str, Any] = {}
        jobs: List[Any] = []
        for obj in _parse_json_stream(merged):
            if not isinstance(obj, dict):
                continue
            if "plan" in obj:
                plan = dict(obj.get("plan") or {})
            if "jobs" in obj:
                jobs = list(obj.get("jobs") or [])
            if isinstance(obj.get("spec"), dict):
                spec_obj = obj["spec"]
            elif spec_obj is None and "name" in obj:
                spec_obj = obj  # spec 边的裸 SPEC 对象
        if jobs:
            plan["flesh"] = jobs
        return json.dumps({"spec": spec_obj, "plan": plan}, ensure_ascii=False)

    def _edge_frame_code(plan_text: str) -> str:
        """逐文件让模型重写，再把结果交给 `assemble` 解固定点。

        输入结构是 `{"spec": ..., "plan": {...}}`，重写任务在 **`plan["plan"]["flesh"]`**。
        曾经这里写成 `plan.get("flesh")` 直接读外层，永远拿到空列表 ⇒ 模型一次都没被调用，
        产物是磁盘源码原样回放（"假通过"）。读错一层比读错一个键更难发现。
        """
        payload = json.loads(plan_text)
        spec_obj = payload.get("spec")
        plan = payload.get("plan") or {}
        parts = dict(base)
        jobs = plan.get("flesh") or []
        rejected: Dict[str, str] = {}
        for job in jobs:
            path = job["path"]
            code = _extract_code(_ask(job["prompt"]), "python")
            if not code.strip():
                rejected[path] = "空产出"
                continue
            # 模型产出的文件先过结构契约检查；不合格就**保留原文件**并记下原因，
            # 不能让坏代码流到 assemble 里把整条链炸掉（也便于事后定位是哪一次生成坏了）。
            #
            # 🔴 只做 `ast.parse` 是**不够**的：实测模型会把 `dag.py` 重写成两行孤立语句
            # （`if models is None and isinstance(...)`，92 字节），把 `llm.py` 重写成
            # 引用未定义 `LLMError`/`json` 的残缺片段（839 字节）。两者 `ast.parse` 都过，
            # 于是被当成"合格"，框架表面自洽、内里被掏空——又是一次假通过。
            # 所以改成**按原文件的结构契约比对**：顶层定义（类/函数/赋值）不许无故消失。
            why = _contract_violation(path, code)
            if why:
                rejected[path] = why
                continue
            parts[path] = code
        _LAST_VERDICT["rejected"] = rejected
        return json.dumps({"files": parts, "spec": spec_obj, "rejected": rejected}, ensure_ascii=False)

    def _edge_manifest(payload: str) -> str:
        data = _parse_json_stream(payload)[-1]
        files = assemble(data["files"], spec=normalize_spec(data["spec"]))
        ns = _probe(files[MAIN_SOURCE])
        ok, why = ns["verify_self_reproduction"]()
        rejected = data.get("rejected") or _LAST_VERDICT.get("rejected") or {}
        _LAST_VERDICT.update(
            {"okay": ok, "reason": why, "manifest": files[FRAME_FILES[0]], "rejected": rejected}
        )
        # result 边要交付的是**整份可运行框架**，不是清单：把全部文件带上，逐字节保留。
        return json.dumps(
            {
                "okay": ok,
                "reason": why,
                "rejected": rejected,
                "files": [{"path": p, "source": files[p]} for p in FRAME_FILES],
            },
            ensure_ascii=False,
        )

    def _edge_result(payload: str) -> str:
        """把多文件产物还原成单份源码：主源原样返回，其余文件内联进 FRAME_MANIFEST 的 b64。

        主源自带自指槽，本身就是自包含的——它内部 `_inline_files()` 会从 BOOTSTRAP_B64
        解出整份框架，所以这里直接给主源即可；校验结论写进 `_LAST_VERDICT` 供调用方查。
        """
        data = _parse_json_stream(payload)[-1]
        files = {item["path"]: item["source"] for item in data["files"]}
        _LAST_VERDICT["okay"] = bool(data.get("okay"))
        _LAST_VERDICT["reason"] = str(data.get("reason") or "")
        _LAST_VERDICT["files"] = files
        return files[MAIN_SOURCE]

    dag.add_edge(["input"], _edge_spec, ["spec_plan"], layer=1)
    dag.add_edge(["input"], _edge_skeleton, ["skeleton_plan"], layer=1)
    dag.add_edge(["input"], _edge_flesh, ["flesh_plan"], layer=1)
    dag.add_edge(["spec_plan", "skeleton_plan", "flesh_plan"], _edge_skeleton_code, ["skeleton_code"], layer=2)
    dag.add_edge(["skeleton_code"], _edge_frame_code, ["frame_code"], layer=3)
    dag.add_edge(["frame_code"], _edge_manifest, ["manifest"], layer=4)
    dag.add_edge(["manifest"], _edge_result, ["result"], layer=5)
    return dag


#: 最近一次 `_edge_result` 的校验结论（`okay` / `reason` / `files`）；供调用方与上层框架查询。
_LAST_VERDICT: Dict[str, Any] = {}


def _parse_json_stream(text: str) -> List[Any]:
    return parse_json_stream(text)


def _merge_plan(spec: Dict[str, Any], raw: str) -> Dict[str, List[str]]:
    """合并模型的签名计划：SPEC 文件只读；public_api 缺谁就从上一版补回同名声明。"""
    proposed: Dict[str, List[str]] = {}
    try:
        data = json.loads(_extract_code(raw, "json"))
        for item in data.get("files", []):
            proposed[str(item.get("path"))] = [str(x) for x in item.get("sig", [])]
    except Exception:
        proposed = {}
    base = framework_sources()
    plan: Dict[str, List[str]] = {}
    for p in FRAME_FILES:
        prev = signature(p, base[p])
        sigs = list(prev) if p in SPEC_FILES else list(proposed.get(p) or prev)
        for name in spec.get("public_api", []):
            if not any(_name_of(s) == name for s in sigs):
                sigs.extend(s for s in prev if _name_of(s) == name)
        plan[p] = sigs
    return plan


def _name_of(sig_line: str) -> str:
    m = re.match(r"^(?:def|class)\s+([A-Za-z_]\w*)", sig_line.strip())
    return m.group(1) if m else sig_line.strip()


def _extract_code(text: str, want: str = "python") -> str:
    """剥掉 markdown 围栏，返回代码/JSON 本体。"""
    t = (text or "").strip()
    m = re.search(r"```(?:python|json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    return t if want == "json" else t + ("" if t.endswith("\n") else "\n")


def _clip(text: str, limit: int = 0) -> str:
    """把源码喂给模型。`limit<=0` 表示**不截断**（默认）。

    原来默认 `limit=9000`，直接把 `dag.py`（27234 字符）砍掉三分之二再让模型
    "重写"——模型看不到后半段，重写出来的自然是残缺片段（实测报
    `'(' was never closed`，契约检查判为不合格并回退）。截断本身就在制造坏输入。
    laptop-8000 上的模型上下文 262144（sensenova 系）到 1048576，最大的
    `bootstrap.py` 也只有八万多字节，整份发过去绰绰有余。
    """
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n# ...(截断)\n"


def bootstrap(
    description: str = "",
    llm: Optional[Any] = None,
    *,
    model_key: str = "laptop-6.8",
    rewrite_files: Sequence[str] = (),
    max_ticks: int = 12,
) -> BootstrapResult:
    """跑一次生成 DAG：描述 → 框架文件集 + 清单 + 自校验结论。"""
    dag = create_generation_dag(llm=llm, model_key=model_key, rewrite_files=rewrite_files)
    dag.send({"tags": ["input"], "content": description, "gen": 0})
    ticks = 0
    while not dag.has_result() and ticks < max_ticks:
        dag.run()
        ticks += 1
    raw = dag.get_result()
    if not raw:
        raise RuntimeError("生成 DAG 没跑出 tag='result'")
    # `result` 边交付的是主源本身（`_edge_result` 已把校验结论放进 `_LAST_VERDICT`），
    # 所以这里不能 json.loads；校验信息从边函数留下的旁路取。
    verdict = dict(_LAST_VERDICT)
    spec_obj = normalize_spec(description)
    files = verdict.get("files") or framework_sources()
    files = assemble(files, spec=spec_obj)
    ok, why = verify_self_reproduction(files[MAIN_SOURCE])
    return BootstrapResult(
        files=files,
        # 清单跟着这一次生成的 spec 走，不能用无参 `manifest()`（它固定读 FRAMEWORK_SPEC）
        manifest_text=_manifest_of_spec(spec_obj),
        okay=ok,
        reason=why,
        context=list(dag.context),
    )
