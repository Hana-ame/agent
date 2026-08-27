#!/usr/bin/env python3
"""Example 3: Human-in-the-Loop (HITL) Workflow with Checkpointing & Resumption.

Demonstrates how a sensitive vertex (e.g. `PaymentGateway`) automatically PAUSES
and snapshots its entire state to SQLite, waiting for an external human review/approval
before resuming and completing downstream execution.

Run:
    python examples/hitl_approval/demo.py
"""

import asyncio
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from framework import (
    Graph,
    CheckpointedExecutor,
    SQLiteStateStore,
    MockAgent,
    VertexState,
)

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def main():
    print(f"\n{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}{CYAN}   v2.0 Demo: Human-in-the-Loop (HITL) Approval & Resume  {RESET}")
    print(f"{BOLD}{CYAN}============================================================{RESET}\n")

    db_path = "/tmp/demo_hitl_checkpoint.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    store = SQLiteStateStore(db_path)

    # Define a 3-step sensitive financial workflow: Order -> RiskGate (PAUSES) -> BankTransfer
    config = {
        "metadata": {"name": "Sensitive_Payout_Pipeline"},
        "vertices": [
            {
                "id": "OrderInitiation",
                "initial_data": [{"channel": "order", "value": {"amount": 50000, "user": "Alice"}}],
            },
            {
                "id": "HighValueRiskApproval",
                # Config-driven Human Gate: pauses when inputs arrive
                "settings": {"require_approval": True},
            },
            {
                "id": "BankTransferExecution",
            },
        ],
        "edges": [
            {
                "id": "e1_eval_risk",
                "source": "OrderInitiation",
                "destination": "HighValueRiskApproval",
                "channel": "order",
                "prompt": "Evaluate payout risk for order",
            },
            {
                "id": "e2_execute_transfer",
                "source": "HighValueRiskApproval",
                "destination": "BankTransferExecution",
                "channel": "order",
                "prompt": "Execute bank transfer",
            },
        ],
    }

    # Step 1: Initial Execution
    print(f"{BOLD}Phase 1: Starting initial workflow execution...{RESET}")
    graph = Graph.from_dict(config)
    executor = CheckpointedExecutor(graph, agents=MockAgent(), store=store, graph_config=config)

    result = await executor.run()
    
    risk_vertex = graph.vertices["HighValueRiskApproval"]
    print(f"  • Result success: {result.success}")
    print(f"  • Risk Gate State: {YELLOW}{risk_vertex.state.value.upper()}{RESET}")
    print(f"  • Bank Transfer State: {MAGENTA}{graph.vertices['BankTransferExecution'].state.value.upper()}{RESET}")
    print(f"  • {BOLD}Execution automatically paused! State snapshot saved to SQLite.{RESET}\n")

    # Step 2: Human Operator Inspects & Approves
    print(f"{BOLD}Phase 2: Human Operator Review (HITL Intervention){RESET}")
    print(f"  ┌────────────────────────────────────────────────────────┐")
    print(f"  │ [PENDING APPROVAL] Vertex: HighValueRiskApproval        │")
    print(f"  │ Details: Payout request of $50,000 to user 'Alice'     │")
    print(f"  └────────────────────────────────────────────────────────┘")
    
    # Simulate a human approving and modifying/enriching the payload
    print(f"  ↳ Operator approves transfer with auth token...")
    await asyncio.sleep(1.0)
    
    # Step 3: Resume Execution from SQLite snapshot
    print(f"\n{BOLD}Phase 3: Resuming workflow from SQLite snapshot...{RESET}")
    
    # Reload fresh graph from the config
    resumed_graph = Graph.from_dict(config)
    
    # Call approve on the paused vertex before resume
    resumed_gate = resumed_graph.vertices["HighValueRiskApproval"]
    resumed_gate.approve({"approved_by": "Compliance_Officer_Bob", "auth_code": "AUTH-9921"})
    print(f"  • Invoked {GREEN}resumed_gate.approve(...){RESET} -> Gate approved and marked {GREEN}READY{RESET}")

    # Resume the executor using the classmethod
    resume_result = await CheckpointedExecutor.resume(
        run_id=executor.run_id,
        graph=resumed_graph,
        agents=MockAgent(),
        store=store,
    )

    print(f"  • Resumed run success: {GREEN}{resume_result.success}{RESET}")
    print(f"  • Final Bank Transfer State: {GREEN}{resumed_graph.vertices['BankTransferExecution'].state.value.upper()}{RESET}")
    
    final_data = await resumed_graph.vertices["BankTransferExecution"].get_all_data()
    print(f"\n{BOLD}Final Data delivered to Bank Transfer:{RESET}")
    for k, v in final_data.items():
        print(f"  • {k}: {v}")
    print()

    # Clean up demo db
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    asyncio.run(main())
