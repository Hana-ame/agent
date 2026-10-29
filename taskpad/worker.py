#!/usr/bin/env python3
"""TaskPad 节点（worker）：从网站认领任务 → 本地执行 → 上传输出与成果文件。

网站只做调度与存储，不执行任何命令。节点脚本负责：
  1. POST /api/runs/claim        认领一个待认领任务（含上游成果：文本内联 / 文件下载地址）
  2. 本地执行命令（相对根目录 --root；engineCwd 提示自动映射到 --root 下）
  3. POST /api/runs/:id/heartbeat 执行期间心跳（同时接收“中止”指令）
  4. POST /api/runs/:id/files/<名> 上传文件型成果
  5. POST /api/runs/:id/submit   上传 stdout / stderr / 退出码，网站落库为 trace 与成果卡

用法：
  python3 worker.py                                    # 连 http://127.0.0.1:8330
  python3 worker.py --server http://192.168.1.10:8330  # 连其它机器上的网站
  python3 worker.py --root /path/to/project            # 命令的相对工作目录
  python3 worker.py --once                             # 只跑一个任务（便于调试）

仅使用 Python 标准库。
"""

import argparse
import fnmatch
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_SERVER = "http://127.0.0.1:8330"


def http_json(url, payload=None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_bytes(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()



def run_iface_tests(tests, cwd, extra_env, port_name, side, port_id):
    """执行接口测试（shell，exit 0 = 通过），返回上报结构列表。"""
    results = []
    for t in tests or []:
        cmd = (t.get("command") or "").strip()
        if not cmd:
            continue
        env = dict(os.environ)
        env.update(extra_env or {})
        try:
            proc = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                                  capture_output=True, timeout=120)
            out = (proc.stdout + proc.stderr).decode("utf-8", "replace")[:4000]
            passed = proc.returncode == 0
            results.append({
                "portId": port_id, "side": side, "testId": t.get("id", ""),
                "name": t.get("name") or cmd[:40], "passed": passed,
                "exitCode": proc.returncode, "output": out
            })
            print("  [test %s/%s] %s -> %s" % (side, port_name, cmd, "PASS" if passed else "FAIL"),
                  flush=True)
        except Exception as e:  # noqa: BLE001
            results.append({
                "portId": port_id, "side": side, "testId": t.get("id", ""),
                "name": t.get("name") or cmd[:40], "passed": False,
                "exitCode": None, "output": "[测试执行异常] %s" % e
            })
    return results

def read_stream(pipe, arr):
    while True:
        chunk = pipe.read(4096)
        if not chunk:
            break
        arr.append(chunk)


def execute(base, root, worker_id, run, heartbeat_sec):
    rid = run["runId"]
    print("[run %s] 认领: %s" % (rid[:8], run.get("taskName")), flush=True)
    print("[run %s] 命令: %s" % (rid[:8], run.get("command")), flush=True)

    cwd = root
    ec = run.get("engineCwd") or "."
    if ec not in (".", "", None):
        cand = os.path.join(root, ec)
        if os.path.isdir(cand):
            cwd = cand

    env = dict(os.environ)
    tmp = tempfile.mkdtemp(prefix="taskpad-in-")
    iface_tests = []
    input_data = {}
    for inp in run.get("inputs") or []:
        kind = inp.get("kind")
        port_id = inp.get("port") or ""
        port_name = inp.get("portName") or port_id
        if kind == "text":
            env[inp["env"]] = inp.get("value", "")
            input_data[port_id] = ("text", inp.get("value", ""))
            print("  [in] %s <- 内联文本 %dB" % (inp["env"], len(inp.get("value", ""))), flush=True)
        elif kind == "file":
            url = base + inp.get("fileUrl", "")
            try:
                data = http_bytes(url)
                fn = os.path.basename(inp.get("fileName") or "input.bin") or "input.bin"
                fp = os.path.join(tmp, fn)
                with open(fp, "wb") as f:
                    f.write(data)
                env[inp["env"]] = fp
                input_data[port_id] = ("file", fp)
                print("  [in] %s <- 下载 %s (%dB)" % (inp["env"], fn, len(data)), flush=True)
            except Exception as e:  # noqa: BLE001
                print("  [in] 下载失败 %s: %s" % (url, e), flush=True)
        else:
            print("  [in] %s 不可用: %s" % (port_name, inp.get("reason")), flush=True)
    # 输入接口测试（数据不达标则不执行任务）
    for inp in run.get("inputs") or []:
        tests = inp.get("tests") or []
        if not tests:
            continue
        port_id = inp.get("port") or ""
        port_name = inp.get("portName") or port_id
        kind, val = input_data.get(port_id, (None, None))
        extra = {"__INPUT": val if kind == "text" else "", "__INPUT_FILE": val if kind == "file" else ""}
        if kind == "text" and val is not None:
            fn = os.path.join(tmp, "input-%s.txt" % port_id)
            with open(fn, "w", encoding="utf-8") as f:
                f.write(val)
            extra["__INPUT_FILE"] = fn
        iface_tests.extend(run_iface_tests(tests, cwd, extra, port_name, "in", port_id))
    in_fail = [t for t in iface_tests if t["side"] == "in" and not t["passed"]]
    if in_fail:
        msg = "\n".join("[接口测试未通过] %s: %s\n%s" % (t["name"], t.get("exitCode"), t.get("output", "")) for t in in_fail)
        print("[run %s] 输入接口测试未通过，跳过执行" % rid[:8], flush=True)
        submit(base, worker_id, rid, 1, "", msg, False, [], iface_tests)
        return

    try:
        proc = subprocess.Popen(
            run["command"],
            shell=True,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:  # noqa: BLE001
        submit(base, worker_id, rid, None, "", "[节点启动失败] %s\n" % e, True, None, [])
        return

    out_chunks, err_chunks = [], []
    t1 = threading.Thread(target=read_stream, args=(proc.stdout, out_chunks), daemon=True)
    t2 = threading.Thread(target=read_stream, args=(proc.stderr, err_chunks), daemon=True)
    t1.start()
    t2.start()

    aborted = False
    last_hb = time.time()
    while proc.poll() is None:
        time.sleep(0.5)
        if time.time() - last_hb >= heartbeat_sec:
            last_hb = time.time()
            try:
                resp = http_json(base + "/api/runs/%s/heartbeat" % rid, {"workerId": worker_id}, timeout=10)
                if resp.get("abort"):
                    aborted = True
                    print("[run %s] 收到中止指令，终止进程" % rid[:8], flush=True)
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
            except urllib.error.HTTPError as e:
                if e.code == 409:
                    # 已被网站回收/其它节点接管，停止执行
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return
                # 其它错误忽略，继续跑
            except Exception:  # noqa: BLE001
                pass

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    t1.join(timeout=3)
    t2.join(timeout=3)

    stdout = b"".join(out_chunks).decode("utf-8", "replace")
    stderr = b"".join(err_chunks).decode("utf-8", "replace")

    # 上传文件型成果 + 收集各端口成果 + 输出接口测试
    uploaded = []
    out_data = {}
    for o in run.get("outputs") or []:
        cap = o.get("capture") or "stdout"
        if cap == "file":
            pattern = o.get("glob") or "*"
            try:
                names = [fn for fn in os.listdir(cwd) if fnmatch.fnmatch(fn, pattern)]
                names.sort(key=lambda fn: os.path.getmtime(os.path.join(cwd, fn)), reverse=True)
                if not names:
                    print("  [out] 端口 %s 未匹配到文件 (%s)" % (o.get("name"), pattern), flush=True)
                    continue
                fn = names[0]
                fp = os.path.join(cwd, fn)
                with open(fp, "rb") as f:
                    data = f.read()
                q = urllib.parse.quote(fn)
                req = urllib.request.Request(base + "/api/runs/%s/files/%s" % (rid, q), data=data, method="POST")
                with urllib.request.urlopen(req, timeout=120):
                    pass
                uploaded.append({"portId": o["id"], "name": fn})
                out_data[o["id"]] = ("file", fp)
                print("  [out] 成果上传: %s (%dB)" % (fn, len(data)), flush=True)
            except Exception as e:  # noqa: BLE001
                print("  [out] 上传失败: %s" % e, flush=True)
        else:
            out_data[o["id"]] = ("text", stdout)
    # 输出接口测试（__RESULT 文本 / __RESULT_FILE 文件）
    for o in run.get("outputs") or []:
        tests = o.get("tests") or []
        if not tests:
            continue
        kind, val = out_data.get(o["id"], (None, None))
        if kind is None:
            continue
        extra = {"__RESULT": val if kind == "text" else "",
                 "__RESULT_FILE": val if kind == "file" else ""}
        if kind == "text":
            fn = os.path.join(tempfile.mkdtemp(prefix="taskpad-out-"), "result.txt")
            with open(fn, "w", encoding="utf-8") as f:
                f.write(val)
            extra["__RESULT_FILE"] = fn
        iface_tests.extend(run_iface_tests(tests, cwd, extra, o.get("name"), "out", o["id"]))

    submit(base, worker_id, rid, None if aborted else proc.returncode, stdout, stderr, aborted, uploaded, iface_tests)
    print(
        "[run %s] 完成 exit=%s aborted=%s out=%dB err=%dB"
        % (rid[:8], proc.returncode, aborted, len(stdout), len(stderr)),
        flush=True,
    )


def submit(base, worker_id, rid, exit_code, stdout, stderr, aborted, uploaded=None, tests=None):
    payload = {
        "workerId": worker_id,
        "exitCode": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "aborted": aborted,
        "files": uploaded or [],
        "tests": tests or [],
    }
    for attempt in range(3):
        try:
            http_json(base + "/api/runs/%s/submit" % rid, payload, timeout=30)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 200:
                return True
            if e.code == 404 and attempt == 2:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser(description="TaskPad 节点 worker")
    ap.add_argument("--server", default=os.environ.get("TASKPAD_SERVER", DEFAULT_SERVER))
    ap.add_argument("--worker-id", default=os.environ.get("TASKPAD_WORKER_ID", ""))
    ap.add_argument("--root", default=".", help="命令相对执行根目录")
    ap.add_argument("--interval", type=float, default=3.0, help="空闲轮询间隔秒")
    ap.add_argument("--heartbeat", type=float, default=5.0, help="心跳间隔秒")
    ap.add_argument("--once", action="store_true", help="只处理一个任务后退出")
    args = ap.parse_args()

    base = args.server.rstrip("/")
    worker_id = args.worker_id or socket.gethostname()
    root = os.path.abspath(args.root)
    print("[worker] id=%s server=%s root=%s" % (worker_id, base, root), flush=True)

    while True:
        try:
            data = http_json(base + "/api/runs/claim", {"workerId": worker_id}, timeout=20)
        except Exception as e:  # noqa: BLE001
            print("[claim error] %s" % e, flush=True)
            time.sleep(max(1.0, args.interval))
            continue
        run = data.get("run")
        if not run:
            if args.once:
                print("[worker] 没有待认领任务，退出", flush=True)
                return
            time.sleep(max(0.5, args.interval))
            continue
        try:
            execute(base, root, worker_id, run, max(1.0, args.heartbeat))
        except Exception as e:  # noqa: BLE001
            print("[exec error] %s" % e, flush=True)
        if args.once:
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
