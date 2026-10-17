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
BOOTSTRAP_SLOT = 46171

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
      "bytes": 12399,
      "b64_len": 5688,
      "sha256": "511262694cb29797330eabcbb1d55a1696a1535af51dbe5d34e25b673112db8d",
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
      "bytes": 29372,
      "b64_len": 11556,
      "sha256": "80b69a0f2b986259bfe753d38489ba0af91c34025c7c76361ba9de902d0620c5",
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
      "bytes": 82687,
      "b64_len": 46171,
      "sha256": "49429ddbd2c85b4d641a76cfb47db351e9b2e1e08cb8709cc8424cd5336994ef",
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


def _solve(main_src: str, manifest_of, code_text: str, self_bytes: int = 0) -> Tuple[int, str]:
    """求槽宽 pad 与最终主源（含把清单里主源那一行的 sha256 收敛到位）。

    两层解耦仍然是基础：`SELF_FILL` 是定长占位符、`BOOTSTRAP_SLOT = pad` 只写数字，
    所以"填充"不是解不动点的难点；难点只剩清单里主源 `bytes`（= 渲染结果自身长度）
    与 `sha256`（= 涂掉自己那一格之后的整文哈希）两个自指量。两者都已变成**一遍可算**：
    - `bytes`：用上一轮的测量值代入（外层 `assemble` 迭代到相等）；
    - `sha256`：涂掉自己那一格 ⇒ `x` 不影响 `f(x)`，代入即收敛。
    """
    body = _render(main_src, manifest_of(32127, "", self_bytes), code_text)
    z = _gzlen(_set_slot(body, 32127))
    # 余量：渲染与最终形态之间的常数差（缩进/位数抖动），几千字节足够且不影响正确性
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
                # 与 tools/fixpoint_v3.py 的写法**必须逐字一致**，否则两边会解出不同 pad
                # （43049 vs 43051 这类），表现为"固化了却仍报需重新固化"。
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

    # 两遍结算：第一遍用占位字节数（位数可能少一位），拿到初步长度；
    # 第二遍用真实字节数 —— 走到这里位宽已稳定，再解一次即自洽（位数若变化会多走一遍）。
    self_bytes = 100000
    src = files[MAIN_SOURCE]
    for _ in range(8):
        pad, src = _solve(
            files[MAIN_SOURCE],
            lambda pad, sha, nbytes: manifest_text(pad, nbytes, sha),
            code_text,
            self_bytes,
        )
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


def _clip(text: str, limit: int = 9000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n# ...(截断)\n"


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
