import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_db import PromptDB
from resolve_prompt import resolve_prompt

db = PromptDB()

questions = [
    # 语文
    # 数学
    # 英语
    # 地理
    # 政治
    # 法律
    # 生物

    "小学语文：'床前明月光'的下一句是什么？这首诗的作者是谁？",
    "小学数学：一个水池，进水管5小时注满，出水管8小时排空。同时打开两管，几小时注满？",
    "小学英语：用英语写一段50字的自我介绍，包含名字、年龄、爱好。",
    "初中地理：为什么中国南方种水稻、北方种小麦？",
    "高中政治：社会主义核心价值观24字是什么？",
    "初中法律：未成年人犯罪需要承担刑事责任吗？请说明年龄界限。",
    "高中生物：光合作用的化学方程式是什么？光反应和暗反应分别在叶绿体的什么部位进行？",
    "初中语文：鲁迅的《朝花夕拾》是什么体裁？收录了多少篇散文？",
    "小学数学：一个三角形三个内角的度数比是1:2:3，这是什么三角形？",
    "高中英语：请用英文解释什么是'定语从句'，并给出两个例句。",
    "世界地理：为什么日本多火山地震？",
    "高中政治：我国的根本政治制度是什么？基本政治制度有哪些？",
    "初中法律：消费者权益保护法规定的'七天无理由退货'适用于哪些商品？",
    "初中生物：人体的消化系统包括哪些器官？食物消化吸收的主要场所是哪里？",
    "高中语文：'落霞与孤鹜齐飞，秋水共长天一色'出自哪篇文章？作者是谁？",
]

model_8b = "siliconflow-cn/Qwen/Qwen3-8B"
model_35_4b = "siliconflow-cn/Qwen/Qwen3.5-4B"

pids_8b = []
pids_35 = []

for q in questions:
    pids_8b.append(db.add(q, agent="Null", model=model_8b))
    pids_35.append(db.add(q, agent="Null", model=model_35_4b))

print(f"Inserted {len(questions)} × 2 = {len(pids_8b) + len(pids_35)} entries")

all_pending = db.list_by_status("pending")
print(f"Total pending: {len(all_pending)}")

results = []
for row in all_pending:
    pid = row["id"]
    agent = row["agent"] or "Null"
    model = row["model"]
    context = row["context"]

    prompt = {"agent": agent, "context": context}
    t0 = time.time()
    try:
        result = resolve_prompt(prompt, db=db, model=model, timeout=300)
        elapsed = time.time() - t0
        db.done(pid, result, {"source": "diverse_batch", "elapsed": round(elapsed, 2)})
        results.append({"id": pid, "model": model.split("/")[-1], "question": context, "answer": result[:200], "elapsed": round(elapsed, 1)})
        print(f"#{pid} {model.split('/')[-1]} [{elapsed:.0f}s] {context[:30]}...")
    except Exception as e:
        db.failed(pid, str(e))
        results.append({"id": pid, "model": model.split("/")[-1], "question": context, "error": str(e)})
        print(f"#{pid} ERROR: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for r in results:
    print(f"#{r['id']} {r['model']}: {r['question'][:50]}")
    print(f"  -> {r.get('answer','ERROR: '+r.get('error',''))[:100]}")
    print()

report = {"timestamp": time.time(), "results": results}
report_path = "reports/diverse_test.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"Report saved to {report_path}")
