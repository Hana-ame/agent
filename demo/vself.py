"""这一代的验收脚本：用本代框架跑自选任务，逐条断言。

运行：python vtest.py      —— 全过返回 0，任一不过返回 1。
"""
import sys

sys.path.insert(0, "..")

from vea_v6 import DAG, SmartMockLLM  # noqa: E402


def _text(x):
    """各代对"处理函数收到什么"的包装不完全一样：有的是字符串，有的是带 .content 的条目。"""
    return x.content if hasattr(x, "content") else str(x)


def parse_line(content):
    """一条单输入边：把日志行切成五个字段，回一行 `k=v`。

    不假设"图会帮我合并多路输入"——切分全在这一个函数里完成。
    """
    line = _text(content)
    # 203.0.113.7 - - [18/Oct/2026:10:00:01 +0800] "GET /api/items?page=2 HTTP/1.1" 200 512
    # 引号里的请求行本身带空格，按空白切会切坏，所以按引号定界切：
    head, _, rest = line.partition('"')
    request, _, tail = rest.partition('"')
    method, path_full = request.split(" ", 1)
    status, nbytes = tail.split()
    return "client_ip=%s method=%s path=%s status=%s bytes=%s" % (
        head.split()[0], method, path_full.split("?")[0], status, nbytes)


def _check_all(fields):
    text = fields.content if hasattr(fields, "content") else str(fields)
    kv = dict(part.split("=", 1) for part in text.split() if "=" in part)
    need = ("client_ip", "method", "path", "status", "bytes")
    miss = [f for f in need if not kv.get(f)]
    if miss:
        raise ValueError("字段没凑齐: " + ",".join(miss) + " | " + text)
    return "READY " + " ".join("%s=%s" % (f, kv[f]) for f in need)


def _summarize(joined):
    text = joined.content if hasattr(joined, "content") else str(joined)
    if not text.startswith("READY"):
        raise ValueError("字段没凑齐: " + text)
    kv = {}
    for part in text.split()[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k] = v
    ok = "yes" if kv.get("status", "").isdigit() and 200 <= int(kv["status"]) < 400 else "no"
    return ("client=%s method=%s path=%s status=%s ok=%s bytes=%s"
            % (kv.get("client_ip", "?"), kv.get("method", "?"), kv.get("path", "?"),
               kv.get("status", "?"), ok, kv.get("bytes", "?")))


def make_dag():
    """三条**单输入**边串成的流水线。

    🔴 为什么一条多输入边都不用：边一旦有多个输入 tag，各代框架对
    "汇聚后怎么调处理函数"的做法并不统一——实测同一组字段，
    磁盘那一代把所有条目按行拼成一条字符串再调，gen1 只给了其中两条
    （`203.0.113.7
512`）。那样验收结论就只对某一代成立，不是"每代都可用"。
    单输入边在三代里行为完全一致，所以这里只用单输入边。
    """
    d = DAG("log_pipeline", llm=SmartMockLLM())
    d.add_edge(["input"], parse_line, ["fields"], layer=1)
    d.add_edge(["fields"], _check_all, ["all_fields"], layer=2)
    d.add_edge(["all_fields"], _summarize, ["result"], layer=3)
    return d


CASES = [
    ('203.0.113.7 - - [18/Oct/2026:10:00:01 +0800] "GET /api/items?page=2 HTTP/1.1" 200 512',
     "client=203.0.113.7 method=GET path=/api/items status=200 ok=yes bytes=512"),
    ('198.51.100.9 - - [18/Oct/2026:10:00:02 +0800] "POST /login HTTP/1.1" 500 78',
     "client=198.51.100.9 method=POST path=/login status=500 ok=no bytes=78"),
]


def main():
    for i, (line, want) in enumerate(CASES, 1):
        d = make_dag()
        d.send({"tags": ["input"], "content": line})
        for _ in range(12):
            if d.has_result():
                break
            d.run()
        got = d.get_result()
        if got != want:
            # 报错文本用 repr，不要用 f-string 里带字面换行（会写坏源码）
            print("case%d FAIL\n  got =%r\n  want=%r" % (i, got, want))
            return 1
        print("case%d OK" % i)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
