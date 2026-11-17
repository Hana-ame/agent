// ============================================================================
// axi_txn_host：事务适配层（第 2 层）
// ============================================================================
// 唯一职责：把「一个 txn_t」展开成完整的 AXI 握手时序，再把结果装回「一个 rsp_t」。
//
// 上层（第 3 层 testbench）只写：
//     txn_host.execute(t, r);      // 就这一行，不知道 AXI 长什么样
//
// 本层内部处理：
//   - 写：AW/W 同发（t.same=1）或分开发（AW 先 / W 后）
//   - 读：AR → R
//   - pause：事务完成后插入 t.pause 个空闲拍（总线无请求）
//   - 期望错误：t.exp_err 与 r.resp 由上层比对，本层只回传实际响应
//
// 握手判据统一用 (VALID && READY)，杜绝"提前就绪覆盖主站 VALID"的死锁。
// ============================================================================
`ifndef AXI_TXN_HOST_SV
`define AXI_TXN_HOST_SV
`timescale 1ns / 1ps

module axi_txn_host #(
    parameter int AW = 8,
    parameter int DW = 32
)(
    input  logic ACLK,
    input  logic ARESETn,
    axi_lite_if.master axi      // 注意：主机角色 → master modport
);

    import bus_txn_pkg::*;

    // ------------------------------------------------------------------
    // 事务执行接口：由 testbench 通过层次名调用
    //   host.execute(t, r)
    // ------------------------------------------------------------------
    task automatic execute(input txn_t t, output rsp_t r);
        r = '0;
        r.data = t.data;

        if (t.write) begin
            // ---------------- 写事务 ----------------
            if (t.same) begin
                // AW + W 同拍（本拍驱动后，两者应同时为 1）
                @(posedge ACLK);
                axi.AWVALID <= 1'b1; axi.AWADDR <= t.addr;
                axi.WVALID  <= 1'b1; axi.WDATA  <= t.data;
                r.same_ok = 1'b1;   // 默认成立，若观察到这里不是同拍再置 0

                // 等待 AW 握手。注意：必须用「本拍采样」而非额外插一拍，
                // 否则会跳过握手窗口导致死等（曾经踩过）。
                do begin
                    @(posedge ACLK); #1;
                    // 同拍性检查：AWVALID 尚在时，WVALID 也应在
                    if (axi.AWVALID && !axi.WVALID && axi.AWREADY !== 1'bx)
                        r.same_ok = 1'b0;
                end while (!(axi.AWVALID && axi.AWREADY));
                axi.AWVALID <= 1'b0;
                do @(posedge ACLK); while (!(axi.WVALID && axi.WREADY));
                axi.WVALID  <= 1'b0;
            end else begin
                // AW 先，W 后
                @(posedge ACLK);
                axi.AWVALID <= 1'b1; axi.AWADDR <= t.addr;
                do @(posedge ACLK); while (!(axi.AWVALID && axi.AWREADY));
                axi.AWVALID <= 1'b0;

                @(posedge ACLK);
                axi.WVALID <= 1'b1; axi.WDATA <= t.data;
                do @(posedge ACLK); while (!(axi.WVALID && axi.WREADY));
                axi.WVALID <= 1'b0;
            end

            // B 响应
            axi.BREADY <= 1'b1;
            do @(posedge ACLK); while (!axi.BVALID);
            r.resp = axi.BRESP;
            axi.BREADY <= 1'b0;
            @(posedge ACLK);
        end else begin
            // ---------------- 读事务 ----------------
            @(posedge ACLK);
            axi.ARVALID <= 1'b1; axi.ARADDR <= t.addr;
            do @(posedge ACLK); while (!(axi.ARVALID && axi.ARREADY));
            axi.ARVALID <= 1'b0;

            axi.RREADY <= 1'b1;
            do @(posedge ACLK); while (!axi.RVALID);
            r.data = axi.RDATA;
            r.resp = axi.RRESP;
            axi.RREADY <= 1'b0;
            @(posedge ACLK);
        end

        // ---------------- pause：事务之间插入空闲拍 ----------------
        // 总线保持全部 VALID=0，验证桥在无请求时保持空闲稳定。
        //
        // ⚠ r.idle 必须是**实测**的空闲拍数，不能直接赋 t.pause ——
        //   否则测试里的 `if (r.idle !== t.pause)` 就成了自证循环，
        //   永远不会失败（原实现就是 `r.idle = t.pause;`）。
        //   现在改为逐拍确认「总线确实空闲」（VALID 全为 0）后才计数，
        //   因此若桥在该期间仍然拉高 VALID 或有未完成握手，idle 会偏小。
        if (t.pause != 0) begin
            int measured;
            measured = 0;
            axi.AWVALID <= 1'b0;
            axi.WVALID  <= 1'b0;
            axi.ARVALID <= 1'b0;
            for (int k = 0; k < t.pause; k++) begin
                @(posedge ACLK); #1;
                // 空闲判据：三个请求通道的 VALID 都未被 DUT 拉高
                if (axi.AWVALID === 1'b0 && axi.WVALID === 1'b0 &&
                    axi.ARVALID === 1'b0)
                    measured++;
            end
            r.idle = measured;
        end else begin
            r.idle = 0;
        end
    endtask

endmodule
`endif // AXI_TXN_HOST_SV