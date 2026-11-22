"""Shared data structures: uniform ToolResult envelope and compiled IR types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_AXIL_SIGNALS = (
    "awaddr", "awvalid", "awready",
    "wdata", "wstrb", "wvalid", "wready",
    "bresp", "bvalid", "bready",
    "araddr", "arvalid", "arready",
    "rdata", "rresp", "rvalid", "rready",
)
OPTIONAL_AXIL_SIGNALS = ("awprot", "arprot")

AXIL_SLAVE_EXPECTED_DIR = {
    "awaddr": "input", "awprot": "input", "awvalid": "input", "awready": "output",
    "wdata": "input", "wstrb": "input", "wvalid": "input", "wready": "output",
    "bresp": "output", "bvalid": "output", "bready": "input",
    "araddr": "input", "arprot": "input", "arvalid": "input", "arready": "output",
    "rdata": "output", "rresp": "output", "rvalid": "output", "rready": "input",
}

REQUIRED_APB_SIGNALS = (
    "psel", "penable", "pwrite", "paddr", "pwdata",
    "pready", "prdata", "pslverr",
)
OPTIONAL_APB_SIGNALS = ("pstrb", "pprot")

APB_SLAVE_EXPECTED_DIR = {
    "psel": "input", "penable": "input", "pwrite": "input",
    "paddr": "input", "pwdata": "input", "pstrb": "input", "pprot": "input",
    "pready": "output", "prdata": "output", "pslverr": "output",
}

APB_MASTER_EXPECTED_DIR = {
    k: ("output" if v == "input" else "input") for k, v in APB_SLAVE_EXPECTED_DIR.items()
}

VALID_ACCESS_MODES = ("RW", "RO", "W1C", "W1S", "RC", "RSVD", "WO")
RESP_CODE_MAP = {"OKAY": 0, "EXOKAY": 1, "SLVERR": 2, "DECERR": 3}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    data: Dict[str, Any]
    diagnostics: List[Dict[str, Any]]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ok_result(tool: str, data: Dict[str, Any], diagnostics: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return ToolResult(ok=True, tool=tool, data=data, diagnostics=diagnostics or []).to_dict()


def err_result(tool: str, message: str, diagnostics: Optional[List[Dict[str, Any]]] = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # data 默认 {} 保持向后兼容；需要携带结构化汇总的工具（如 lint_tb 违规时
    # 仍要返回 error_count/warning_count/findings）可显式传 data。
    return ToolResult(ok=False, tool=tool, data=data if data is not None else {}, diagnostics=diagnostics or [], error=message).to_dict()


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ClockSpecIR:
    port: str
    freq_mhz: float


@dataclass(frozen=True)
class ResetSpecIR:
    port: str
    active_low: bool
    sync: bool


@dataclass(frozen=True)
class OutOfScopePortIR:
    port: str
    direction: str
    width: int
    tie_off: Optional[str] = None
    observe_only: bool = False


@dataclass(frozen=True)
class AxiLiteBoundaryIR:
    addr_width: int
    data_width: int
    role: str
    port_map: Dict[str, str]

    @property
    def strb_width(self) -> int:
        return self.data_width // 8

    @property
    def data_mask(self) -> int:
        return (1 << self.data_width) - 1


@dataclass(frozen=True)
class ApbBoundaryIR:
    addr_width: int
    data_width: int
    role: str
    port_map: Dict[str, str]
    wait_cycles: int = 0


@dataclass(frozen=True)
class BitFieldIR:
    name: str
    msb: int
    lsb: int
    access: str
    reset_value: int = 0
    description: str = ""

    @property
    def width(self) -> int:
        return self.msb - self.lsb + 1


@dataclass(frozen=True)
class BoundaryIR:
    module_name: str
    instance_hierarchy: Optional[str]
    rtl_files: Tuple[str, ...]
    parameters: Dict[str, Any]
    clock: ClockSpecIR
    reset: ResetSpecIR
    axil: AxiLiteBoundaryIR
    out_of_scope_ports: Tuple[OutOfScopePortIR, ...]
    apb: Optional[ApbBoundaryIR] = None


@dataclass(frozen=True)
class ProtocolRuleIR:
    check_valid_stability: bool
    check_reset_outputs: bool
    max_resp_cycles: int
    allow_unaligned_addr: bool


@dataclass(frozen=True)
class RegisterRuleIR:
    rule_id: str
    name: str
    offset: int
    reset_value: int
    access: str
    write_mask: int
    read_mask: int
    fields: Tuple[BitFieldIR, ...] = ()


@dataclass(frozen=True)
class IllegalAddrRuleIR:
    rule_id: str
    offset: int
    expected_resp: str


@dataclass(frozen=True)
class StressProfileIR:
    random_backpressure: bool
    iterations: int
    seed: int


@dataclass(frozen=True)
class QualityGatesIR:
    min_rtl_coverage: float = 90.0
    min_mutation_score: float = 80.0
    require_lint_clean: bool = True
    require_suite_identity_check: bool = True


@dataclass(frozen=True)
class TestPlanIR:
    protocol_rules: ProtocolRuleIR
    register_rules: Tuple[RegisterRuleIR, ...]
    illegal_addr_rules: Tuple[IllegalAddrRuleIR, ...]
    stress_profile: StressProfileIR
    quality_gates: QualityGatesIR = QualityGatesIR()


@dataclass(frozen=True)
class CompiledUnitIR:
    boundary: BoundaryIR
    test_plan: TestPlanIR
