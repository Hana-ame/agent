#!/bin/bash
# resolve_prompt 测试脚本
# 用法: bash test_resolve_prompt.sh

set -e
cd "$(dirname "$0")"

PASS=0
FAIL=0
TOTAL=0

assert() {
    TOTAL=$((TOTAL+1))
    local desc="$1" input="$2" expected="$3"
    actual=$(python3 -c "
import sys; sys.path.insert(0,'.')
from resolve_prompt import resolve_prompt
from prompt_db import PromptDB
db = PromptDB()
print(resolve_prompt($input, db=db, model='opencode/mimo-v2.5-free', timeout=30))
" 2>&1)
    if echo "$actual" | grep -qF "$expected"; then
        echo "  ✅ $desc"
        PASS=$((PASS+1))
    else
        echo "  ❌ $desc"
        echo "     期望包含: $expected"
        echo "     实际输出: $actual"
        FAIL=$((FAIL+1))
    fi
}

# ── 准备 DB ──
python3 -c "
import sys; sys.path.insert(0,'.')
from prompt_db import PromptDB
db = PromptDB()
with db._conn() as conn:
    conn.execute('DELETE FROM prompts')
    conn.commit()
"

echo "=============================================="
echo "  resolve_prompt 测试"
echo "=============================================="

# ── Case 1: 纯文本 ──
echo ""
echo "--- Case 1: 纯文本 ---"
assert "1+1等于几" '"1+1等于几？请直接回答数字"' "2"

# ── Case 2: agent + text ──
echo ""
echo "--- Case 2: agent + text ---"
assert "2+3等于几" '{"agent":"Null","context":"2+3等于几？请直接回答数字"}' "5"

# ── Case 3: agent + list ──
echo ""
echo "--- Case 3: agent + list ---"
assert "水果+蔬菜" '{"agent":"Null","context":["请说出一种水果","请说出一种蔬菜"]}' ""

# ── Case 4: int 引用（有 response）──
echo ""
echo "--- Case 4: int 引用（有 response）---"
python3 -c "
import sys; sys.path.insert(0,'.')
from prompt_db import PromptDB
db = PromptDB()
pid = db.add('你最喜欢什么颜色？', agent='Null', model='mimo-v2.5-free')
db.done(pid, '蓝色是我最喜欢的颜色。')
print(pid)
" > /tmp/pid4.txt
PID4=$(cat /tmp/pid4.txt)
assert "int $PID4 有 response" "$PID4" "蓝色是我最喜欢的颜色。"

# ── Case 5: int 引用（无 response，从 context 推理）──
echo ""
echo "--- Case 5: int 引用（无 response，从 context 推理）---"
python3 -c "
import sys; sys.path.insert(0,'.')
from prompt_db import PromptDB
db = PromptDB()
pid1 = db.add('你最喜欢什么颜色？', agent='Null', model='mimo-v2.5-free')
db.done(pid1, '蓝色是我最喜欢的颜色。')
pid2 = db.add('你最喜欢什么食物？', agent='Null', model='mimo-v2.5-free')
db.done(pid2, '面条是我最喜欢的食物。')
pid3 = db.add('总结上面两个回答', agent='Null', model='mimo-v2.5-free', context=[pid1, pid2])
print(f'{pid1} {pid2} {pid3}')
" > /tmp/pid5.txt
PID5_1=$(awk '{print $1}' /tmp/pid5.txt)
PID5_2=$(awk '{print $2}' /tmp/pid5.txt)
PID5_3=$(awk '{print $3}' /tmp/pid5.txt)
assert "int $PID5_3 从 context 推理" "$PID5_3" "蓝色是我最喜欢的颜色。"
assert "int $PID5_3 包含食物" "$PID5_3" "面条是我最喜欢的食物。"

# ── Case 6: int 引用（无 response 无 context）──
echo ""
echo "--- Case 6: int 引用（无 response 无 context）---"
python3 -c "
import sys; sys.path.insert(0,'.')
from prompt_db import PromptDB
db = PromptDB()
pid = db.add('空记录', agent='Null', model='mimo-v2.5-free')
print(pid)
" > /tmp/pid6.txt
PID6=$(cat /tmp/pid6.txt)
RESULT6=$(python3 -c "
import sys; sys.path.insert(0,'.')
from resolve_prompt import resolve_prompt
from prompt_db import PromptDB
db = PromptDB()
r = resolve_prompt($PID6, db=db, model='opencode/mimo-v2.5-free', timeout=30)
print('EMPTY' if not r else 'HAS_CONTENT')
" 2>&1)
if [ "$RESULT6" = "EMPTY" ]; then
    echo "  ✅ int $PID6 空返回"
    PASS=$((PASS+1))
else
    echo "  ❌ int $PID6 应为空，实际: $RESULT6"
    FAIL=$((FAIL+1))
fi

# ── Case 7: list 混合 str + int ──
echo ""
echo "--- Case 7: list 混合 str + int ---"
assert "str+int 混合" "{\"agent\":\"Null\",\"context\":[\"你喜欢的水果是\",$PID5_1,\"你喜欢的食物是\",$PID5_2]}" ""

# ── Case 8: 嵌套 Prompt + int ──
echo ""
echo "--- Case 8: 嵌套 Prompt + int ---"
assert "嵌套 int" "{\"agent\":\"Null\",\"context\":[\"根据以下信息回答：\",{\"agent\":\"Null\",\"context\":[$PID5_1,$PID5_2]}]}" ""

# ── 结果 ──
echo ""
echo "=============================================="
echo "  结果: $PASS/$TOTAL 通过, $FAIL 失败"
echo "=============================================="
