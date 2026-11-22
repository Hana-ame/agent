"""位域测试生成辅助（列表构建，避免转义问题）"""


def parse_bits(bs):
    s = str(bs)
    if ":" in s:
        hi, lo = s.split(":")
        return int(hi), int(lo)
    v = int(s)
    return v, v


def build_field_test(reg, aw, dw):
    """生成一个位域寄存器的测试代码行"""
    L = []
    name = reg["name"]
    off = int(str(reg["offset"]), 0)
    off_s = f"{aw}'h{off:02X}"
    L.append(f'        // ===== {name} 位域测试 @0x{off:02X} =====')

    for f in reg.get("fields", []):
        hi, lo = parse_bits(f["bits"])
        width = hi - lo + 1
        mask = ((1 << width) - 1) << lo
        mask_s = f"{dw}'h{mask:08X}"
        a = f["attr"]
        nm = f["name"]
        tag = f"{name}.{nm}"

        if a == "RW":
            L.append(f'        // {tag} [{hi}:{lo}] RW: 写满该域应可读回')
            L.append(f'        write_reg({off_s}, {mask_s});')
            L.append(f'        read_reg({off_s}, d);')
            L.append(f'        checks++;')
            L.append(f'        if ((d & {mask_s}) !== {mask_s}) begin')
            L.append(f'            $display("  [FLD] FAIL: {tag} RW 写满回读 d=%h mask=%h", d, {mask_s});')
            L.append(f'            errors++;')
            L.append(f'        end')
            L.append(f'        write_reg({off_s}, \'0);')
        elif a == "RO":
            L.append(f'        // {tag} [{hi}:{lo}] RO: 写应被忽略')
            L.append(f'        read_reg({off_s}, d);')
            L.append(f'        base = d & {mask_s};')
            L.append(f'        write_reg({off_s}, {mask_s});')
            L.append(f'        read_reg({off_s}, d);')
            L.append(f'        checks++;')
            L.append(f'        if ((d & {mask_s}) !== base) begin')
            L.append(f'            $display("  [FLD] FAIL: {tag} RO 被写改动 base=%h now=%h", base, d);')
            L.append(f'            errors++;')
            L.append(f'        end')
        elif a == "W1C":
            L.append(f'        // {tag} [{hi}:{lo}] W1C: 写 1 应清零')
            # 关键：先复位恢复位域初值，否则前面测试已把该位域写成 0，
            # 「清零」与「保持」无差异，变异无法被发现
            L.append(f'        do_reset();')
            L.append(f'        read_reg({off_s}, d);   // 复位后应有初值')
            L.append(f'        base = d & {mask_s};')
            L.append(f'        checks++;')
            L.append(f'        if (base === \'0) begin')
            L.append(f'            $display("  [FLD] FAIL: {tag} W1C 复位后初值为0，清零不可观测");')
            L.append(f'            errors++;')
            L.append(f'        end')
            L.append(f'        write_reg({off_s}, {mask_s});')
            L.append(f'        read_reg({off_s}, d);')
            L.append(f'        checks++;')
            L.append(f'        if ((d & {mask_s}) !== \'0) begin')
            L.append(f'            $display("  [FLD] FAIL: {tag} W1C 写 1 未清零 d=%h", d);')
            L.append(f'            errors++;')
            L.append(f'        end')
        elif a == "W1S":
            L.append(f'        // {tag} [{hi}:{lo}] W1S: 写 1 置位')
            L.append(f'        write_reg({off_s}, \'0);   // 先清零（通过 W1C 不可行，直接读初值）')
            L.append(f'        write_reg({off_s}, {mask_s});   // 写 1 应置位')
            L.append(f'        read_reg({off_s}, d);')
            L.append(f'        checks++;')
            L.append(f'        if ((d & {mask_s}) !== {mask_s}) begin')
            L.append(f'            $display("  [FLD] FAIL: {tag} W1S 未置位 d=%h", d);')
            L.append(f'            errors++;')
            L.append(f'        end')
        elif a == "RC":
            L.append(f'        // {tag} [{hi}:{lo}] RC: 读后清零')
            L.append(f'        do_reset();   // 恢复 RC 位域初值')
            L.append(f'        read_reg({off_s}, d);   // 第一次读（应为初值）')
            L.append(f'        base = d & {mask_s};')
            L.append(f'        checks++;')
            L.append(f'        if (base === \'0) begin')
            L.append(f'            $display("  [FLD] FAIL: {tag} RC 复位后初值为0，读清零不可观测");')
            L.append(f'            errors++;')
            L.append(f'        end')
            L.append(f'        read_reg({off_s}, d);   // 第二次读应为 0')
            L.append(f'        checks++;')
            L.append(f'        if ((d & {mask_s}) !== \'0) begin')
            L.append(f'            $display("  [FLD] FAIL: {tag} RC 读后未清零 d=%h", d);')
            L.append(f'            errors++;')
            L.append(f'        end')
        elif a == "RSVD":
            L.append(f'        // {tag} [{hi}:{lo}] RSVD: 写忽略、读 0')
            L.append(f'        write_reg({off_s}, {mask_s});')
            L.append(f'        read_reg({off_s}, d);')
            L.append(f'        checks++;')
            L.append(f'        if ((d & {mask_s}) !== \'0) begin')
            L.append(f'            $display("  [FLD] FAIL: {tag} RSVD 非 0 d=%h", d);')
            L.append(f'            errors++;')
            L.append(f'        end')

    # 多域独立性：所有 RW 域同时写满
    rw_mask = 0
    has_rw = False
    for f in reg.get("fields", []):
        if f["attr"] == "RW":
            hi, lo = parse_bits(f["bits"])
            rw_mask |= ((1 << (hi - lo + 1)) - 1) << lo
            has_rw = True
    if has_rw:
        rw_s = f"{dw}'h{rw_mask:08X}"
        L.append(f'        // {name} 多 RW 域同时写满，验证互不干扰')
        L.append(f'        write_reg({off_s}, {rw_s});')
        L.append(f'        read_reg({off_s}, d);')
        L.append(f'        checks++;')
        L.append(f'        if ((d & {rw_s}) !== {rw_s}) begin')
        L.append(f'            $display("  [FLD] FAIL: {name} 多域写满 d=%h mask=%h", d, {rw_s});')
        L.append(f'            errors++;')
        L.append(f'        end')
        L.append(f'        write_reg({off_s}, \'0);')

    return L
