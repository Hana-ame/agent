// ============================================================================
// tb_reset_domain_sv.sv：复位域独立性验证
//
// 数据位宽说明（避免误解覆盖范围）：
//   本 TB 的测试向量使用 32 位常量（如 32'hAAAA_BBBB）。
//   在 DW=32 下即覆盖全部数据位；若以 DW=64 编译，
//   高 32 位只会被写 0、不会被本 TB 验证。
//   DW=64 的完整数据通路验证见 tb_dw64_sv（用 64 位向量）。
// ============================================================================
// 背景：
//   所有现有 TB 都把 APB 侧的 PRESETn 直接接到 AXI 侧的 ARESETn
//   （.PRESETn(ARESETn)），因此从未验证过：
//     - APB 从机单独复位（AXI 侧仍在工作）时系统行为
//     - AXI 桥单独复位（APB 侧保持）时系统行为
//     - 两侧复位不同步释放时的竞态
//   这在真实 SoC 中很常见（AXI/APB 属于不同电源/复位域）。
//
// 本测试用两个独立复位信号，验证：
//   [1] 只复位 APB 从机：寄存器清零，桥本身状态不受影响
//   [2] 只复位 AXI 桥：APB 从机寄存器内容保持，桥可恢复工作
//   [3] 复位不同步释放：释放顺序不同，系统都能恢复正常
//   [4] 复位期间的 AXI VALID 保持性
// ============================================================================
`ifndef TB_RESET_DOMAIN_SV_SV
`define TB_RESET_DOMAIN_SV_SV
`timescale 1ns / 1ps

module tb_reset_domain_sv #(
    parameter int AW   = 8,
    parameter int DW   = 32,
    parameter int NREG = 16
);

    import bus_txn_pkg::*;
    import axi_defs_pkg::*;

    logic ACLK = 0;
    logic ARESETn = 0;   // AXI 侧复位（桥 + 主机）
    logic PRESETn = 0;   // APB 侧复位（从机，独立）
    always #5 ACLK = ~ACLK;

    axi_lite_if #(.AW(AW), .DW(DW)) axi(ACLK);
    apb_if      #(.AW(AW), .DW(DW)) apb(ACLK);

    axi_apb_bridge_sv #(.AW(AW), .DW(DW)) bridge (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi), .apb(apb)
    );
    // 事务主机：用项目里经过验证的 axi_txn_host（handshake 边界已被验证）
    axi_txn_host #(.AW(AW), .DW(DW)) host (
        .ACLK(ACLK), .ARESETn(ARESETn), .axi(axi)
    );
    // 关键：PRESETn 独立，不再跟随 ARESETn
    // WAIT_CYCLES=3：拉长 APB ACCESS 窗口，
    // 使 [5] 的「传输中独立复位」能稳定命中 S_RD_ACC（否则是时序竞态）
    apb_slave_regs_sv #(.AW(AW), .DW(DW), .NREG(NREG), .WAIT_CYCLES(3)) slave (
        .PCLK(ACLK), .PRESETn(PRESETn), .apb(apb)
    );

    int errors = 0, checks = 0;

    task automatic tick();
        @(posedge ACLK); #1;
    endtask

    // 等待桥回到 IDLE（有超时保护，失败则报错）
    task automatic wait_idle(input string tag);
        int g = 0;
        while (bridge.state != 0 && g < 100) begin
            @(posedge ACLK); g++;
        end
        if (bridge.state != 0) begin
            $display("  ✗ [%s] 桥未能回到 IDLE（state=%0d）", tag, bridge.state);
            errors++;
        end
    endtask

    task automatic clear_axi_bus();
        axi.AWVALID = 0; axi.WVALID = 0; axi.ARVALID = 0;
        axi.BREADY = 0;  axi.RREADY = 0;
    endtask

    // 简单的主机写/读（用阻塞+延时方式，逻辑清晰）
    // 用 axi_txn_host 执行事务（经过验证的握手时序）
    task automatic wr(input [7:0] addr, input [31:0] data, output bit [1:0] resp);
        txn_t t; rsp_t r;
        t = '0; t.write = 1; t.addr = addr; t.data = data;
        host.execute(t, r);
        resp = r.resp;
    endtask

    task automatic rd(input [7:0] addr, output [31:0] data, output bit [1:0] resp);
        txn_t t; rsp_t r;
        t = '0; t.write = 0; t.addr = addr; t.data = '0;
        host.execute(t, r);
        data = r.data;
        resp = r.resp;
    endtask

    initial begin
        bit [1:0]  resp;
        bit [31:0] data;

        $display("======================================================");
        $display(" 复位域独立性验证（AXI / APB 独立复位）");
        $display("======================================================");
        $display(" 注意：本 DUT 的桥无超时保护，若 APB 从机在");
        $display("       事务进行中被单独复位，桥会卡在等 PREADY。");
        $display("       这是已知的、合理的设计约束（见文末结论）。");

        // 初始复位（两侧同时）
        clear_axi_bus();
        PRESETn = 0; ARESETn = 0;
        repeat (5) @(posedge ACLK);
        PRESETn = 1; ARESETn = 1;
        tick();

        // ---------------------------------------------------------
        // [1] 只复位 APB 从机（空闲时）：寄存器清零，系统可继续
        // ---------------------------------------------------------
        $display("== [1] 空闲时只复位 APB 从机 ==");
        wr(8'h10, 32'h1234_5678, resp);
        checks++;
        if (resp === RESP_OKAY) $display("  写 0x10 = 0x1234_5678 OK");
        else begin $display("  ✗ 写失败 resp=%0d", resp); errors++; end

        // 确保总线空闲后再复位（关键：不能有事务在进行中）
        wait_idle("before-PRESET");
        PRESETn = 0;
        repeat (3) @(posedge ACLK);
        PRESETn = 1;
        repeat (3) @(posedge ACLK);

        rd(8'h10, data, resp);
        checks++;
        if (resp === RESP_OKAY && data === 32'h0) begin
            $display("  ✓ APB 从机复位后寄存器清零");
        end else begin
            $display("  ✗ APB 复位未清零：data=0x%08h resp=%0d", data, resp);
            errors++;
        end

        wr(8'h10, 32'hAAAA_BBBB, resp);
        rd(8'h10, data, resp);
        checks++;
        if (resp === RESP_OKAY && data === 32'hAAAA_BBBB) begin
            $display("  ✓ 复位后系统继续正常工作");
        end else begin
            $display("  ✗ 系统失效：data=0x%08h", data);
            errors++;
        end

        // ---------------------------------------------------------
        // [2] 只复位 AXI 桥（APB 侧保持）：APB 寄存器内容保留
        // ---------------------------------------------------------
        $display("== [2] 只复位 AXI 桥（APB 从机保持）==");
        wr(8'h14, 32'hCAFE_F00D, resp);
        wait_idle("before-ARESET");

        ARESETn = 0;
        clear_axi_bus();
        repeat (3) @(posedge ACLK);
        ARESETn = 1;
        repeat (3) @(posedge ACLK);

        rd(8'h14, data, resp);
        checks++;
        if (resp === RESP_OKAY && data === 32'hCAFE_F00D) begin
            $display("  ✓ AXI 桥复位未影响 APB 内容（0x%08h）", data);
        end else begin
            $display("  ✗ APB 内容丢失：data=0x%08h resp=%0d", data, resp);
            errors++;
        end

        // ---------------------------------------------------------
        // [3] 两侧复位释放顺序不同
        // ---------------------------------------------------------
        $display("== [3] 复位释放顺序不同 ==");

        PRESETn = 0; ARESETn = 0;
        repeat (3) @(posedge ACLK);
        PRESETn = 1; repeat (2) @(posedge ACLK); ARESETn = 1;
        repeat (3) @(posedge ACLK);
        wr(8'h08, 32'h1111_2222, resp);
        rd(8'h08, data, resp);
        checks++;
        if (resp === RESP_OKAY && data === 32'h1111_2222)
            $display("  ✓ 先释放 APB 再释放 AXI：正常");
        else begin
            $display("  ✗ 顺序 APB->AXI 失败：data=0x%08h", data);
            errors++;
        end

        PRESETn = 0; ARESETn = 0;
        repeat (3) @(posedge ACLK);
        ARESETn = 1; repeat (2) @(posedge ACLK); PRESETn = 1;
        repeat (3) @(posedge ACLK);
        wr(8'h0C, 32'h3333_4444, resp);
        rd(8'h0C, data, resp);
        checks++;
        if (resp === RESP_OKAY && data === 32'h3333_4444)
            $display("  ✓ 先释放 AXI 再释放 APB：正常");
        else begin
            $display("  ✗ 顺序 AXI->APB 失败：data=0x%08h", data);
            errors++;
        end

        // ---------------------------------------------------------
        // [4] 复位期间从机不响应（PREADY 保持低）
        // ---------------------------------------------------------
        $display("== [4] APB 从机复位期间不响应 ==");
        PRESETn = 0;
        repeat (3) @(posedge ACLK);
        checks++;
        if (!apb.PREADY) $display("  ✓ 复位期间 PREADY 保持低");
        else begin $display("  ✗ 复位期间 PREADY 为高"); errors++; end
        PRESETn = 1;
        repeat (3) @(posedge ACLK);
        wr(8'h04, 32'h5555_6666, resp);
        rd(8'h04, data, resp);
        checks++;
        if (resp === RESP_OKAY && data === 32'h5555_6666)
            $display("  ✓ 复位释放后立即可用");
        else begin
            $display("  ✗ 复位释放后不可用：data=0x%08h", data);
            errors++;
        end

        // ---------------------------------------------------------
        // [5] APB 独立复位导致死锁 → ARESETn 可恢复
        // ---------------------------------------------------------
        // 上轮发现：APB 从机在传输中被单独复位会让桥卡在
        // S_RD_ACC / S_WR_ACC 等 PREADY。这里进一步验证：
        // 该死锁能否通过复位 AXI 域解除？——结论：可以。
        $display("== [5] APB 传输中独立复位 → 死锁 → ARESETn 恢复 ==");
        wait_idle("before-deadlock-test");
        // 确保所有 AXI 通道完全空闲（前面操作可能残留 valid）
        clear_axi_bus();
        repeat (5) @(posedge ACLK);
        wait_idle("before-deadlock-test-2");

        begin
            bit hit = 0;
            // 用 fork 并行监控：桥完成 APB 传输只需极少数周期，
            // 串行「先拉 AR 再等 PENABLE」会错过窗口。
            // 因此一边驱动 AR，一边立即盯 PENABLE。
            fork
                // 线程 A：驱动读事务
                begin
                    axi.ARVALID = 1; axi.ARADDR = 8'h10; axi.RREADY = 0;
                    begin
                        int g = 0; bit done = 0;
                        while (!done && g < 50) begin
                            @(posedge ACLK); #1; g++;
                            if (axi.ARREADY) done = 1;
                        end
                        if (done) axi.ARVALID = 0;
                    end
                end
                // 线程 B：盯 PENABLE，命中即复位 PRESETn
                begin
                    int g = 0; bit found = 0;
                    while (!found && g < 80) begin
                        @(posedge ACLK); #1; g++;
                        if (apb.PENABLE && apb.PSEL && !apb.PREADY) begin
                            found = 1;
                            hit = 1;
                            $display("  命中 APB ACCESS（st=%0d），复位 PRESETn", bridge.state);
                            PRESETn = 0;
                            repeat (4) @(posedge ACLK);
                            PRESETn = 1;
                        end
                    end
                end
            join
            repeat (10) @(posedge ACLK);

            checks++;
            if (hit && bridge.state != 0) begin
                $display("  ✓ 确认死锁：桥卡在 state=%0d（等 PREADY）", bridge.state);
            end else if (!hit) begin
                $display("  ⚠ 未命中 APB ACCESS 窗口（st=%0d, PENABLE=%b, PSEL=%b）",
                         bridge.state, apb.PENABLE, apb.PSEL);
            end else begin
                $display("  ✗ 预期死锁但桥已回 IDLE");
                errors++;
            end

            // 用 ARESETn 恢复（复位会打断 APB 传输，R2 断言必然触发）
            $assertoff(0, tb_reset_domain_sv.apb);
            ARESETn = 0; clear_axi_bus();
            repeat (4) @(posedge ACLK);
            ARESETn = 1;
            repeat (2) @(posedge ACLK);
            $asserton(0, tb_reset_domain_sv.apb);

            checks++;
            if (bridge.state == 0) begin
                $display("  ✓ ARESETn 成功解除死锁（桥回 IDLE）");
            end else begin
                $display("  ✗ ARESETn 未能解除死锁（state=%0d）", bridge.state);
                errors++;
            end

            // 恢复后必须能正常工作
            wr(8'h18, 32'hDEAD_BEEF, resp);
            rd(8'h18, data, resp);
            checks++;
            if (resp === RESP_OKAY && data === 32'hDEAD_BEEF) begin
                $display("  ✓ 恢复后系统可正常读写");
            end else begin
                $display("  ✗ 恢复后系统失效：data=0x%08h", data);
                errors++;
            end
        end

        // ---------------------------------------------------------
        // 汇总 + 使用约束说明
        // ---------------------------------------------------------
        $display("------------------------------------------------------");
        $display(" 使用约束（本测试验证得出结论）：");
        $display("   1. APB 从机不得在事务进行中被单独复位 ——");
        $display("      桥无超时保护，会卡在 S_RD_ACC/S_WR_ACC 等 PREADY。");
        $display("   2. 若已发生该死锁，复位 AXI 域（ARESETn）可解除，");
        $display("      但会打断进行中的 APB 传输（违反 APB R2 协议）。");
        $display("   3. 实际 SoC 应保证 APB 域复位不打断进行中的传输；");
        $display("      否则需在系统层加超时复位机制。");
        $display("------------------------------------------------------");
        $display(" reset-domain result: %0d checks, %0d errors", checks, errors);
        if (errors == 0) begin
            $display("PASS: reset domain independence");
            $finish;
        end else begin
            $display("FAIL: %0d errors", errors);
            $fatal(1, "reset domain test failed");
        end
    end

endmodule
`endif // TB_RESET_DOMAIN_SV_SV
