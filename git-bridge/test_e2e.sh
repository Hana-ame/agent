#!/usr/bin/env bash
# End-to-end test for git-bridge (Node) using a local bare remote — no GitHub.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
trap 'kill "$SRV" 2>/dev/null || true; rm -rf "$TMP"' EXIT
echo "tmp=$TMP"

ORIGIN="$TMP/origin.git"
REPO="$TMP/repo"
PORT=18765

git init --bare -q -b main "$ORIGIN"
git clone -q "$ORIGIN" "$REPO"
git -C "$REPO" config user.name tester
git -C "$REPO" config user.email tester@x
printf 'hello\n' > "$REPO/file.txt"
git -C "$REPO" add -A
git -C "$REPO" commit -q -m init
git -C "$REPO" push -q -u origin main

node "$HERE/server.js" --repo "$REPO" --port "$PORT" --base main \
    --ledger "$TMP/ledger.jsonl" > "$TMP/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 60); do
    grep -q "listening on" "$TMP/server.log" && break
    sleep 0.2
done
echo "=== server.log (head) ==="; sed -n '1,12p' "$TMP/server.log"

ok=0; fail=0
assert_eq() {
    if [ "$2" = "$3" ]; then echo "  PASS $1"; ok=$((ok+1));
    else echo "  FAIL $1  (expected [$2] got [$3])"; fail=$((fail+1)); fi
}

B="http://127.0.0.1:$PORT"

echo "=== /health ==="
curl -sS "$B/health"; echo

echo "=== commit #1 (new branch feat-1) ==="
printf '%s\n' '--- a/file.txt' '+++ b/file.txt' '@@ -1 +1 @@' '-hello' '+hello world' > "$TMP/p1.patch"
curl -sS -X POST "$B/commit?branch=feat-1&message=add%20world" --data-binary @"$TMP/p1.patch"
echo
assert_eq "origin/feat-1 file.txt" "hello world" "$(git -C "$ORIGIN" show feat-1:file.txt)"

echo "=== commit #2 (chained on feat-1) ==="
printf '%s\n' '--- a/file.txt' '+++ b/file.txt' '@@ -1 +1 @@' '-hello world' '+hello world!' > "$TMP/p2.patch"
curl -sS -X POST "$B/commit?branch=feat-1&message=bang" --data-binary @"$TMP/p2.patch"
echo
assert_eq "origin/feat-1 after #2" "hello world!" "$(git -C "$ORIGIN" show feat-1:file.txt)"
assert_eq "origin/feat-1 commit count" "3" "$(git -C "$ORIGIN" rev-list --count feat-1)"

echo "=== commit #3 (new file) ==="
printf '%s\n' '--- /dev/null' '+++ b/new.txt' '@@ -0,0 +1 @@' '+new content' > "$TMP/p3.patch"
curl -sS -X POST "$B/commit?branch=feat-new&message=add%20file" --data-binary @"$TMP/p3.patch"
echo
assert_eq "origin/feat-new new.txt" "new content" "$(git -C "$ORIGIN" show feat-new:new.txt)"

echo "=== 6 concurrent POSTs to 6 different branches ==="
pids=()
for i in 1 2 3 4 5 6; do
  printf '%s\n' '--- a/file.txt' '+++ b/file.txt' '@@ -1 +1 @@' "-hello" "+hello from $i" > "$TMP/c$i.patch"
  curl -sS -X POST "$B/commit?branch=conc-$i&message=c$i" --data-binary @"$TMP/c$i.patch" > "$TMP/c$i.out" &
  pids+=($!)
done
wait "${pids[@]}"
for i in 1 2 3 4 5 6; do echo "  conc-$i -> $(git -C "$ORIGIN" show conc-$i:file.txt)"; done
all_ok=1
for i in 1 2 3 4 5 6; do
  [ "$(git -C "$ORIGIN" show conc-$i:file.txt 2>/dev/null)" = "hello from $i" ] || all_ok=0
done
assert_eq "all 6 concurrent branches pushed" "1" "$all_ok"

echo "=== bad patch (expect 400) ==="
printf '%s\n' '--- a/missing.txt' '+++ b/missing.txt' '@@ -1 +1 @@' '-nope' '+y' > "$TMP/bad.patch"
code=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "$B/commit?branch=brBad&message=bad" --data-binary @"$TMP/bad.patch")
echo "http=$code"
assert_eq "bad patch http" "400" "$code"

echo "=== noop patch (expect committed:false) ==="
printf '%s\n' '--- a/file.txt' '+++ b/file.txt' '@@ -1 +1 @@' '-hello' '+hello' > "$TMP/none.patch"
curl -sS -X POST "$B/commit?branch=brNone&message=none" --data-binary @"$TMP/none.patch"; echo

echo "=== async POST + poll ==="
id=$(curl -sS -X POST "$B/commit?branch=feat-async&message=async&async=1" --data-binary @"$TMP/p1.patch" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
echo "id=$id"
sleep 1
st=$(curl -sS "$B/commits/$id" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status"))')
echo "status=$st"
assert_eq "async job reached pushed" "pushed" "$st"

echo "=== /commits ==="
curl -sS "$B/commits?limit=4" | python3 -c 'import sys,json;d=json.load(sys.stdin);print("count",d["count"]);[print(" ",c["branch"],c["status"],c.get("short","")) for c in d["commits"]]'

echo "=== /queue ==="
curl -sS "$B/queue"; echo

echo "=== server.log (tail) ==="; tail -n 25 "$TMP/server.log"
echo "=== RESULT: pass=$ok fail=$fail ==="
[ "$fail" = "0" ]
