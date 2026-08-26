import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_db import PromptDB
from resolve_prompt import resolve_prompt

db = PromptDB()

# 63 questions covering more subjects, to bring total to ~180
qs = [
    "物理：牛顿三大定律是什么？",
    "化学：水的化学式是什么？由什么元素组成？",
    "历史：中华人民共和国是哪一年成立的？",
    "音乐：五线谱上的高音谱号又叫什么谱号？",
    "美术：三原色是哪三种颜色？",
    "体育：奥运会几年举办一次？",
    "计算机：CPU的中文名称是什么？",
    "天文：太阳系有几大行星？",
    "经济：什么是供求关系？",
    "哲学：'我思故我在'是谁说的？",
    "逻辑：如果所有的A都是B，所有的B都是C，那么所有的A都是C吗？",
    "心理学：什么是'巴甫洛夫的狗'实验？",
    "社会学：什么是'马太效应'？",
    "医学：人体有多少块骨头？",
    "军事：'三十六计'中'走为上计'是第几计？",
    "文学：《红楼梦》的作者是谁？",
    "数学：圆周率π的前五位小数是多少？",
    "英语：'apple'和'banana'哪个是水果？用英语回答",
    "地理：中国最长的河流是什么？",
    "政治：我国的国家性质是什么？",
    "法律：宪法规定我国公民的基本权利有哪些？",
    "生物：DNA的中文名称是什么？",
    "语文：'但愿人长久，千里共婵娟'出自谁的作品？",
    "历史：第一次世界大战爆发的导火索是什么？",
    "化学：pH值等于7时溶液呈什么性？",
    "物理：光在真空中的传播速度是多少？",
    "音乐：贝多芬的第五交响曲又叫什么？",
    "计算机：二进制数1010转换成十进制是多少？",
    "天文：地球的卫星是什么？",
    "经济：GDP的中文含义是什么？",
    "医学：疫苗的作用原理是什么？",
    "体育：乒乓球起源于哪个国家？",
    "逻辑：'所有的鸟都会飞，企鹅是鸟，所以企鹅会飞'这个推理正确吗？",
    "地理：中国的面积在世界上排第几？",
    "历史：唐朝的开国皇帝是谁？",
    "生物：人体最大的器官是什么？",
    "物理：为什么天空是蓝色的？",
    "化学：铁生锈是物理变化还是化学变化？",
    "数学：等腰三角形的两个底角有什么关系？",
    "语文：'三人行，必有我师焉'出自哪本书？",
    "英语：用英语写出12个月份的名称",
    "政治：我国的国歌是什么？",
    "法律：什么是正当防卫？",
    "计算机：什么是人工智能？",
    "天文：什么是黑洞？",
    "地理：长江三峡是哪三个峡谷的总称？",
    "历史：鸦片战争发生在哪一年？",
    "生物：人体细胞中有多少对染色体？",
    "物理：什么是相对论？谁提出的？",
    "化学：空气的主要成分是什么？",
    "音乐：什么是交响乐？",
    "数学：勾股定理的内容是什么？",
    "语文：'举头望明月'的下一句是什么？",
    "英语：什么是现在进行时？举例说明",
    "地理：中国的四大高原是哪些？",
    "政治：什么是人民代表大会制度？",
    "法律：什么是知识产权？",
    "生物：什么是光合作用？",
    "计算机：什么是云计算？",
    "历史：秦始皇统一六国是在哪一年？",
    "物理：声音在空气中传播的速度大约是多少？",
    "化学：什么是催化剂？",
    "数学：什么是质数？举一个例子",
]

model_8b = "siliconflow-cn/Qwen/Qwen3-8B"
model_35 = "siliconflow-cn/Qwen/Qwen3.5-4B"

pids = []
for q in qs:
    pids.append(db.add(q, agent="Null", model=model_8b))
    pids.append(db.add(q, agent="Null", model=model_35))

print(f"Added {len(pids)} entries")
print(f"Running...")

pending = db.list_by_status("pending")
print(f"Total pending: {len(pending)}")

for row in pending:
    pid = row["id"]
    model = row["model"]
    context = row["context"]
    prompt = {"agent": "Null", "context": context}
    t0 = time.time()
    try:
        r = resolve_prompt(prompt, db=db, model=model, timeout=300)
        db.done(pid, r)
    except Exception as e:
        db.failed(pid, str(e)[:100])

total = len(db.list_all())
print(f"\nTotal DB entries now: {total}")
