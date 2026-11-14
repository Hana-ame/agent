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
            L.append(f'        write_reg({off_s}, {mask_s});')
            L.append(f'        read_reg({off_s}, d);')
            L.append(f'        checks++;')
            L.append(f'        if ((d & {mask_s}) !== \'0) begin')
            L.append(f'            $display("  [FLD] FAIL: {tag} W1C 写 1 未清零 d=%h", d);')
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
