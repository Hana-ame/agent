"""
位域 RTL 生成辅助（独立函数，全部用列表逐行构建，避免 f-string 转义问题）
"""


def parse_bits(bs):
    """'31:24' -> (31, 24); '3' -> (3, 3)"""
    s = str(bs)
    if ":" in s:
        hi, lo = s.split(":")
        return int(hi), int(lo)
    v = int(s)
    return v, v


def build_field_write_func(reg_name, bitfields, dw):
    """生成一个寄存器的位域写变换函数（返回代码行列表）"""
    L = []
    fn = f"field_write_{reg_name}"
    L.append(f"    // {reg_name} 位域写变换（逐位域按属性合并）")
    L.append(f"    function automatic logic [DW-1:0] {fn}(")
    L.append("        input logic [DW-1:0] old_v,")
    L.append("        input logic [DW-1:0] wr_v")
    L.append("    );")
    L.append("        logic [DW-1:0] nv;")
    L.append("        nv = old_v;")

    for f in bitfields:
        hi, lo = parse_bits(f["bits"])
        width = hi - lo + 1
        mask = ((1 << width) - 1) << lo
        mask_s = f"{dw}'h{mask:08X}"
        a = f["attr"]
        nm = f["name"]
        L.append(f"        // {nm} bits[{hi}:{lo}] {a}")
        if a == "RW":
            L.append(f"        nv = (nv & ~{mask_s}) | (wr_v & {mask_s});")
        elif a == "W1C":
            L.append(f"        nv = (nv & ~{mask_s}) | (old_v & ~wr_v & {mask_s});")
        elif a == "W1S":
            # 写 1 置位：写 1 的位变 1，写 0 的位保持
            L.append(f"        nv = (nv & ~{mask_s}) | ((old_v | wr_v) & {mask_s});")
        elif a in ("RO", "RC"):
            # RO / RC：写忽略（RC 副作用在读路径）
            L.append(f"        nv = (nv & ~{mask_s}) | (old_v & {mask_s});")
        else:
            # RSVD：忽略写
            L.append(f"        nv = (nv & ~{mask_s}) | (old_v & {mask_s});")

    L.append("        return nv;")
    L.append("    endfunction")
    L.append("")
    return L


def build_field_dispatch(field_regs, dw):
    """生成按索引分派到位域函数的 case（返回代码行列表）"""
    L = []
    L.append("    // ---- 位域寄存器写分派（按索引）----")
    L.append("    function automatic logic [DW-1:0] write_dispatch(")
    L.append("        input int            idx,")
    L.append("        input logic [DW-1:0] old_v,")
    L.append("        input logic [DW-1:0] wr_v")
    L.append("    );")
    L.append("        case (idx)")
    for reg in field_regs:
        idx = int(str(reg["offset"]), 0) // (dw // 8)
        L.append(f"            {idx}: return field_write_{reg['name']}(old_v, wr_v);")
    L.append("            default: return write_transform_word(old_v, wr_v, attr_of(idx));")
    L.append("        endcase")
    L.append("    endfunction")
    L.append("")
    return L


def build_field_read_clear(reg_name, bitfields, dw):
    """生成一个位域寄存器的读清零函数（列表构建）
       仅清 RC 属性的位域，其他位域保持不变
    """
    L = []
    fn = f"field_read_clear_{reg_name}"
    rc_fields = [f for f in bitfields if f["attr"] == "RC"]

    L.append(f"    // {reg_name} 位域读清零（仅清 RC 位域）")
    L.append(f"    function automatic logic [DW-1:0] {fn}(")
    L.append("        input logic [DW-1:0] cur_v")
    L.append("    );")
    L.append("        logic [DW-1:0] nv;")
    L.append("        nv = cur_v;")
    if not rc_fields:
        L.append("        // 本寄存器无 RC 位域，保持原值")
    for f in rc_fields:
        hi, lo = parse_bits(f["bits"])
        width = hi - lo + 1
        mask = ((1 << width) - 1) << lo
        mask_s = f"{dw}'h{mask:08X}"
        L.append(f"        nv = nv & ~{mask_s};   // {f['name']} [{hi}:{lo}] RC 读清零")
    L.append("        return nv;")
    L.append("    endfunction")
    L.append("")
    return L


def build_read_clear_dispatch(field_regs, dw):
    """生成按索引分派到位域读清零函数的 case"""
    L = []
    L.append("    // ---- 位域寄存器读清零分派 ----")
    L.append("    function automatic logic [DW-1:0] read_clear_dispatch_fld(")
    L.append("        input int            idx,")
    L.append("        input logic [DW-1:0] cur_v")
    L.append("    );")
    L.append("        case (idx)")
    for reg in field_regs:
        idx = int(str(reg["offset"]), 0) // (dw // 8)
        L.append(f"            {idx}: return field_read_clear_{reg['name']}(cur_v);")
    L.append("            default: return read_clear_word(cur_v, attr_of(idx));")
    L.append("        endcase")
    L.append("    endfunction")
    L.append("")
    return L
