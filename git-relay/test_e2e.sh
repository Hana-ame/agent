#!/usr/bin/env bash
#
# test_e2e.sh — full protocol test for git-relay against a local fake upstream.
#
# Covers: clone, fetch, push (with ref audit), whitelist enforcement,
# read-only rules, endpoint allowlisting, anonymous access, ledger contents.
#
# Usage:  bash test_e2e.sh
# Needs:  git, node, bash. No network, no credentials.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
NODE="${NODE:-node}"
UP_PORT="${UP_PORT:-9900}"
RELAY_PORT="${RELAY_PORT:-9901}"
WORK="$(mktemp -d 2>/dev/null || echo "/tmp/gitrelay-test.$$")"
ROOT="$WORK/upstream"
LOG="$WORK/log"
PASS=0
FAIL=0

mkdir -p "$ROOT" "$LOG"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { PASS=$((PASS+1)); printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
check(){ if [ "$1" = "$2" ]; then ok "$3 ($1)"; else bad "$3 (got '$1', want '$2')"; fi; }

cleanup() {
  [ -n "${UP_PID:-}" ] && kill "$UP_PID" 2>/dev/null
  [ -n "${RELAY_PID:-}" ] && kill "$RELAY_PID" 2>/dev/null
  # keep the work dir on failure so the reason is inspectable
  if [ "$FAIL" -eq 0 ]; then rm -rf "$WORK"; else printf '\nwork dir kept: %s\n' "$WORK"; fi
}
trap cleanup EXIT

# git must not send these localhost requests through a user's global proxy
NOPROXY=(-c "http.http://127.0.0.1:$RELAY_PORT/.proxy=")

# --------------------------------------------------------------------------
say "prepare bare repos"
# --------------------------------------------------------------------------
for r in acme/widget acme/secret; do
  mkdir -p "$ROOT/$r"
  git init --bare -q -b main "$ROOT/$r.git" || exit 1
done

SEED="$WORK/seed"
mkdir -p "$SEED"
(
  cd "$SEED" || exit 1
  git init -q -b main
  git config user.email seed@test
  git config user.name seed
  printf 'hello from upstream\n' > README.md
  mkdir -p src
  printf 'const a = 1\n' > src/a.js
  git add -A
  git commit -q -m "initial commit"
  git push -q "$ROOT/acme/widget.git" main
  git checkout -q -b dev
  printf 'dev only\n' > DEV.md
  git add -A && git commit -q -m "dev branch"
  git push -q "$ROOT/acme/widget.git" dev
  git checkout -q main
) || { echo "seed failed"; exit 1; }
ok "widget.git has main + dev"

# --------------------------------------------------------------------------
say "start fake upstream + relay"
# --------------------------------------------------------------------------
"$NODE" "$HERE/test-upstream.js" "$ROOT" "$UP_PORT" > "$LOG/upstream.log" 2>&1 &
UP_PID=$!

cat > "$WORK/allow.txt" <<'EOF'
# comment line
acme/widget
acme/secret:ro
EOF

"$NODE" "$HERE/server.js" \
  --port "$RELAY_PORT" --upstream "http://127.0.0.1:$UP_PORT" \
  --allow-file "$WORK/allow.txt" \
  --ledger "$WORK/ledger.jsonl" --quiet > "$LOG/relay.log" 2>&1 &
RELAY_PID=$!

for i in $(seq 1 40); do
  curl -sf -o /dev/null "http://127.0.0.1:$RELAY_PORT/_relay/status" && break
  sleep 0.25
done
if curl -sf -o /dev/null "http://127.0.0.1:$RELAY_PORT/_relay/status"; then
  ok "relay is up"
else
  bad "relay did not start"; cat "$LOG/relay.log"; exit 1
fi

# --------------------------------------------------------------------------
say "clone through the relay (anonymous)"
# --------------------------------------------------------------------------
cd "$WORK" || exit 1
if git "${NOPROXY[@]}" clone -q "http://127.0.0.1:$RELAY_PORT/acme/widget.git" c1 2>"$LOG/clone.err"; then
  ok "clone succeeded"
else
  bad "clone failed: $(cat "$LOG/clone.err")"
fi

if [ -f c1/README.md ] && grep -q "hello from upstream" c1/README.md; then
  ok "worktree content matches upstream"
else
  bad "worktree content wrong"
fi

if git -C c1 branch -r 2>/dev/null | grep -q "origin/dev"; then
  ok "remote branch origin/dev advertised"
else
  bad "origin/dev missing"
fi

# clone without the .git suffix — git clients do this
if git "${NOPROXY[@]}" clone -q "http://127.0.0.1:$RELAY_PORT/acme/widget" c1b 2>"$LOG/clone2.err"; then
  ok "clone without .git suffix works"
else
  bad "clone without .git failed: $(cat "$LOG/clone2.err")"
fi

# --------------------------------------------------------------------------
say "fetch an update made upstream"
# --------------------------------------------------------------------------
(
  cd "$SEED" || exit 1
  printf 'more upstream work\n' >> README.md
  git add -A && git commit -q -m "upstream moves on"
  git push -q "$ROOT/acme/widget.git" main
)
if git -C c1 "${NOPROXY[@]}" pull -q --ff-only 2>"$LOG/pull.err"; then
  if grep -q "more upstream work" c1/README.md; then
    ok "pull picked up the upstream commit"
  else
    bad "pull ran but content is stale"
  fi
else
  bad "pull failed: $(cat "$LOG/pull.err")"
fi

# --------------------------------------------------------------------------
say "push through the relay (no credentials at all)"
# --------------------------------------------------------------------------
(
  cd "$WORK/c1" || exit 1
  git config user.email relay-test@local
  git config user.name relay-test
  mkdir -p feature
  printf 'pushed through git-relay\n' > feature/note.md
  git add -A
  git commit -q -m "feat: add a file via relay"
  git "${NOPROXY[@]}" push -q origin HEAD:refs/heads/feature-x 2>"$LOG/push.err"
) && ok "push accepted" || bad "push failed: $(cat "$LOG/push.err")"

# the commit must exist in the *upstream* bare repo, not just locally
UP_SHA=$(git -C "$ROOT/acme/widget.git" rev-parse --short refs/heads/feature-x 2>/dev/null)
if [ -n "$UP_SHA" ]; then
  ok "upstream now has feature-x @ $UP_SHA"
else
  bad "feature-x missing upstream"
fi

if git -C "$ROOT/acme/widget.git" ls-tree -r --name-only refs/heads/feature-x 2>/dev/null | grep -qx "feature/note.md"; then
  ok "pushed file is present in upstream tree"
else
  bad "pushed file missing in upstream tree"
fi

# --------------------------------------------------------------------------
say "audit ledger"
# --------------------------------------------------------------------------
if [ -f "$WORK/ledger.jsonl" ]; then
  n=$(grep -c '"endpoint":"/git-receive-pack"' "$WORK/ledger.jsonl" 2>/dev/null || true)
  n=${n:-0}
  [ "$n" -ge 1 ] && ok "ledger has $n receive-pack record(s)" || bad "no receive-pack record in ledger"
  if grep -q 'refs/heads/feature-x' "$WORK/ledger.jsonl"; then
    ok "ledger captured the pushed ref name"
  else
    bad "ledger did not capture ref names"
  fi
else
  bad "ledger file missing"
fi

# --------------------------------------------------------------------------
say "whitelist enforcement"
# --------------------------------------------------------------------------
code=$(curl -s -o /dev/null -w '%{http_code}' \
  "http://127.0.0.1:$RELAY_PORT/othercorp/private.git/info/refs?service=git-upload-pack")
check "$code" "404" "unlisted repo is invisible (404)"

# --path-as-is keeps curl from normalising ".." away before it leaves
code=$(curl -s --path-as-is -o /dev/null -w '%{http_code}' \
  "http://127.0.0.1:$RELAY_PORT/acme/../acme/widget.git/info/refs?service=git-upload-pack")
check "$code" "404" "path traversal is rejected"

code=$(curl -s --path-as-is -o /dev/null -w '%{http_code}' \
  "http://127.0.0.1:$RELAY_PORT/acme/%2e%2e/widget.git/info/refs?service=git-upload-pack")
check "$code" "404" "percent-encoded traversal is rejected"

# the bare repo exists upstream, but it is not on the whitelist
mkdir -p "$ROOT/othercorp"
git init --bare -q -b main "$ROOT/othercorp/private.git"
code=$(curl -s -o /dev/null -w '%{http_code}' \
  "http://127.0.0.1:$RELAY_PORT/othercorp/private.git/info/refs?service=git-upload-pack")
check "$code" "404" "repo that exists upstream is still blocked"

# --------------------------------------------------------------------------
say "read-only rule"
# --------------------------------------------------------------------------
code=$(curl -s -o /dev/null -w '%{http_code}' \
  "http://127.0.0.1:$RELAY_PORT/acme/secret.git/git-receive-pack")
check "$code" "403" "push endpoint refused on :ro repo"

code=$(curl -s -o /dev/null -w '%{http_code}' \
  "http://127.0.0.1:$RELAY_PORT/acme/secret.git/info/refs?service=git-upload-pack")
check "$code" "200" "clone endpoint still open on :ro repo"

# --------------------------------------------------------------------------
say "endpoint allowlisting"
# --------------------------------------------------------------------------
code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$RELAY_PORT/acme/widget.git/HEAD")
check "$code" "404" "/HEAD (dumb protocol) is not proxied"

code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$RELAY_PORT/acme/widget.git/objects/info/packs")
check "$code" "404" "/objects/... is not proxied"

code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$RELAY_PORT/acme/widget.git")
check "$code" "404" "bare repo root is not proxied"

# --------------------------------------------------------------------------
say "control surface"
# --------------------------------------------------------------------------
body=$(curl -s "http://127.0.0.1:$RELAY_PORT/_relay/status")
echo "$body" | grep -q '"pushes"' && ok "/_relay/status reports counters" || bad "/_relay/status broken"

# --------------------------------------------------------------------------
say "read-only mode"
# --------------------------------------------------------------------------
kill "$RELAY_PID" 2>/dev/null; wait "$RELAY_PID" 2>/dev/null
"$NODE" "$HERE/server.js" \
  --port "$RELAY_PORT" --upstream "http://127.0.0.1:$UP_PORT" \
  --allow acme/widget --read-only \
  --ledger "$WORK/ledger2.jsonl" --quiet > "$LOG/relay2.log" 2>&1 &
RELAY_PID=$!
for i in $(seq 1 40); do
  curl -sf -o /dev/null "http://127.0.0.1:$RELAY_PORT/_relay/status" && break
  sleep 0.25
done
code=$(curl -s -o /dev/null -w '%{http_code}' \
  "http://127.0.0.1:$RELAY_PORT/acme/widget.git/git-receive-pack")
check "$code" "403" "--read-only refuses push globally"

# --------------------------------------------------------------------------
printf '\n\033[1m%d passed, %d failed\033[0m\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
