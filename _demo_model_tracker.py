import model_tracker as fm
import db
from pathlib import Path
import subprocess

# 从 opencode 模型列表抓真实 model 名（取前几个做演示）
result = subprocess.run(["opencode", "models"], capture_output=True, text=True, timeout=15)
real_models = [m.strip() for m in result.stdout.strip().split("\n") if m.strip()]
print(f"共 {len(real_models)} 个模型，取前 3 个演示\n")

# 临时 DB
db.DB_PATH = Path("/tmp/test_demo_real.db")
if db.DB_PATH.exists():
    db.DB_PATH.unlink()

m1, m2, m3 = real_models[0], real_models[1], real_models[2]

conn = fm._get_conn()
conn.execute("INSERT INTO models (model, provider) VALUES (?, ?)", (m1, "opencode"))
conn.execute("INSERT INTO models (model, provider) VALUES (?, ?)", (m2, "opencode"))
conn.execute("INSERT INTO models (model, provider) VALUES (?, ?)", (m3, "nvidia"))
conn.execute("INSERT INTO usage (model, calls, successes, failures) VALUES (?, ?, ?, ?)", (m1, 10, 8, 2))
conn.commit()
conn.close()

print("=== get_stats() 所有模型 ===")
print(fm.get_stats())

print(f"\n=== get_stats('{m1}') 指定模型 ===")
print(fm.get_stats(m1))

print("\n=== get_stats('noop') 不存在 ===")
print(fm.get_stats("noop"))

print(f"\n=== record_call('{m2}', success=True) ===")
fm.record_call(m2, success=True)
print(fm.get_stats(m2))

print(f"\n=== record_call('{m2}', success=False) ===")
fm.record_call(m2, success=False)
print(fm.get_stats(m2))

print(f"\n=== record_call('{m3}', success=True) ===")
fm.record_call(m3, success=True)
print(fm.get_stats(m3))

print("\n=== list_free_models() ===")
print(fm.list_free_models())
