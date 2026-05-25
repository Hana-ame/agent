#!/usr/bin/env python3
"""Board 666 巡检脚本：获取最新帖子，按 ts 降序排列，识别未处理需求并生成报告"""

import json
import sys
import os
from datetime import datetime

def fetch_board_data():
    """使用 moonchan.py 获取 Board 666 数据"""
    import subprocess
    script_path = "/home/lumin/.claude/skills/moonchan-forum/scripts/moonchan.py"
    result = subprocess.run(
        [sys.executable, script_path, "list", "666", "--pn", "0"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return json.loads(result.stdout)

def sort_posts(data):
    """按 ts 降序排列主帖和回复"""
    if not data:
        return data
    
    # 1. 主帖数组按 ts 降序
    data.sort(key=lambda x: x['ts'], reverse=True)
    
    # 验证主帖排序
    assert data[0]['ts'] == max(p['ts'] for p in data), "主帖排序验证失败"
    
    # 2. 每个帖子的回复列表按 ts 降序
    for post in data:
        if 'list' in post and post['list']:
            post['list'].sort(key=lambda x: x['ts'], reverse=True)
            # 验证回复排序
            if post['list']:
                assert post['list'][0]['ts'] == max(r['ts'] for r in post['list']), \
                    f"回复排序验证失败: no={post['no']}"
    
    return data

def check_unprocessed(data):
    """查找未处理的需求（无 Loop666 回复的帖子）"""
    unprocessed = []
    for p in data:
        has_loop666 = any(r.get('n') == 'Loop666' for r in p.get('list', []))
        if not has_loop666:
            unprocessed.append(p)
    return unprocessed

def classify_request(post):
    """判断帖子是否为代码/指令需求"""
    txt = post['txt'].strip()
    
    # 非需求判断
    non_request_indicators = [
        '可能没什么需要做的',
        '没什么需要做的',
    ]
    for ind in non_request_indicators:
        if ind in txt:
            return 'non_request', '非代码需求：无需操作'
    
    # 纯链接（上传文件等）
    if txt.startswith('http') and '\n' not in txt:
        return 'upload', '文件上传/链接'
    
    # 代码需求
    code_indicators = ['```python', '```py', 'def ', 'import ', 'class ', 'timeout', 'retry']
    if any(ind in txt for ind in code_indicators):
        return 'code', '代码需求'
    
    # 指令/任务需求
    instruction_indicators = [
        'ssh', 'add', '修改', '创建', '修复', '执行', '检查',
        'which branch', 'git', '汇报', '回复'
    ]
    if any(ind in txt.lower() for ind in instruction_indicators):
        return 'instruction', '指令/任务需求'
    
    # 抱怨/骂人
    if any(c in txt for c in ['操', '傻逼', '死']):
        return 'rant', '非需求：抱怨/情绪表达'
    
    return 'unknown', '未分类'

def main():
    print("=" * 60)
    print("Board 666 巡检报告")
    print(f"巡检时间: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # 1. 获取数据
    print("\n[1/4] 获取 Board 666 数据...")
    data = fetch_board_data()
    if not data:
        print("❌ 获取数据失败")
        return
    
    print(f"✅ 获取到 {len(data)} 个帖子")
    
    # 2. 排序
    print("\n[2/4] 按 ts 降序排列...")
    data = sort_posts(data)
    latest = data[0]
    print(f"✅ 排序完成")
    print(f"   最新帖子: no={latest['no']} ts={latest['ts']}")
    print(f"   内容: {latest['txt'][:80]}...")
    
    # 3. 检查未处理需求
    print("\n[3/4] 检查未处理需求...")
    unprocessed = check_unprocessed(data)
    
    if not unprocessed:
        print("✅ 所有帖子均已处理（均有 Loop666 回复）")
    else:
        print(f"⚠️  发现 {len(unprocessed)} 个无 Loop666 回复的帖子:")
        for p in unprocessed:
            has_auto = any(r.get('n') == 'Auto666' for r in p.get('list', []))
            cat, reason = classify_request(p)
            status = "Auto666 已处理" if has_auto else "完全未处理"
            need = "需要代码" if cat == 'code' else ("需要操作" if cat == 'instruction' else "无需操作")
            print(f"  - no={p['no']} [{cat}] {status} ({need}): {p['txt'][:60]}")
    
    # 4. 列出所有帖子的处理状态
    print("\n[4/4] 所有帖子状态一览（按时间降序）:")
    print(f"{'no':<8} {'ts':<22} {'Loop666':<10} {'Auto666':<10} {'内容摘要'}")
    print("-" * 80)
    for p in data:
        has_loop = any(r.get('n') == 'Loop666' for r in p.get('list', []))
        has_auto = any(r.get('n') == 'Auto666' for r in p.get('list', []))
        txt_short = p['txt'][:50].replace('\n', ' ')
        print(f"{p['no']:<8} {p['ts']:<22} {'✅' if has_loop else '❌':<10} {'✅' if has_auto else '❌':<10} {txt_short}")
    
    print("\n" + "=" * 60)
    
    # 检查是否有代码需求需要处理
    code_requests = [p for p in unprocessed if classify_request(p)[0] == 'code']
    if code_requests:
        print(f"\n⚠️  发现 {len(code_requests)} 个未处理的代码需求:")
        for p in code_requests:
            print(f"  no={p['no']}: {p['txt'][:100]}")
    else:
        print("\n✅ 当前 Board 666 没有新的未处理代码需求。")
        print("   所有历史指令均已由 Auto666 或 Loop666 执行完毕。")
    
    print("=" * 60)
    return data

if __name__ == '__main__':
    main()
