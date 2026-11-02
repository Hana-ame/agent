// gitproxy — a local HTTPS smart-HTTP reverse proxy to GitHub + a patch/PR bridge.
//
// Listen on :8000 (behind a tunnel -> https://wsl-8000.moonchan.xyz).
//
// What it does:
//  1. git smart-HTTP reverse proxy:
//     git clone https://<host>/<owner>/<repo>[.git]          (branch + depth 1 work natively)
//     git push  https://<host>/<owner>/<repo>[.git] <branch>
//     It forwards info/refs, git-upload-pack, git-receive-pack to github.com,
//     injecting the gh token as auth. Agents need NO credentials.
//  2. Force-push protection: for git-receive-pack, the leading '+' (force) flag
//     is stripped from ref-update commands whose ref already exists on the
//     remote (old-sha != 0). GitHub then rejects any non-fast-forward to an
//     existing branch. New branches and fast-forwards are unaffected.
//  3. POST /commit (or POST /<owner>/<repo>/commit):
//     body = unified diff ; ?branch=&message=&base=&pr=
//     apply -> commit -> push ; optionally open a PR via gh.
//
// Run:   gitproxy --addr :8000 --allow Hana-ame/agent
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

var zeroSHA = strings.Repeat("0", 40)

type App struct {
	addr     string
	dataDir  string
	tmpDir   string // temp dir for patch files (default os.TempDir())
	base     string
	policy   *branchPolicy // protected/allowed branch policy (from --protect/--protect-file)
	allow    []string
	apiToken string // optional bearer protecting POST API

	tokenMu    sync.Mutex
	tokCache   string
	tokCacheAt time.Time

	cloneMu sync.Mutex // serialize local git operations (single-worker queue)
}

var shaRe = regexp.MustCompile(`^[0-9a-f]{40}$`)
var shaLikeRe = regexp.MustCompile(`^[0-9a-fA-F]{7,40}$`)

func main() {
	var allowList, allowFileList arrayFlag
	var protectList, protectFileList arrayFlag
	a := &App{}
	flag.StringVar(&a.addr, "addr", ":8000", "listen address")
	flag.StringVar(&a.dataDir, "data", "./data", "directory for local clones")
	flag.StringVar(&a.tmpDir, "tmp-dir", "", "temp dir for patch files (default: OS temp dir, i.e. /tmp on Linux, %TEMP% on Windows)")
	flag.StringVar(&a.base, "base", "main", "default base branch for new branches / PR base")
	flag.Var(&allowList, "allow", "allowed repo (owner/repo), repeatable; supports * glob and @file (default Hana-ame/agent)")
	flag.Var(&allowFileList, "allow-file", "file with allowed repos, one per line (repeatable; blank lines and # comments ignored)")
	flag.Var(&protectList, "protect", "protected branch globs (repeatable; * = one segment, ** = across '/'; 'none' disables; default = the --base branch)")
	flag.Var(&protectFileList, "protect-file", "branch policy file with [protect]/[allow] sections, one glob per line (repeatable)")
	flag.StringVar(&a.apiToken, "api-token", "", "if set, POST /commit requires Authorization: Bearer <token>")
	flag.Parse()
	var resolved []string
	for _, v := range allowList {
		if strings.HasPrefix(v, "@") {
			lines, err := readAllowFile(strings.TrimPrefix(v, "@"))
			if err != nil {
				logf("allow @file error: %v", err)
				os.Exit(2)
			}
			resolved = append(resolved, lines...)
		} else {
			resolved = append(resolved, v)
		}
	}
	for _, f := range allowFileList {
		lines, err := readAllowFile(f)
		if err != nil {
			logf("--allow-file error: %v", err)
			os.Exit(2)
		}
		resolved = append(resolved, lines...)
	}
	if len(resolved) == 0 {
		resolved = []string{"Hana-ame/agent"}
	}
	a.allow = dedupeStrings(resolved)
	pol, perr := loadBranchPolicy(protectFileList, protectList, a.base)
	if perr != nil {
		logf("branch policy error: %v", perr)
		os.Exit(2)
	}
	a.policy = pol
	if a.tmpDir == "" {
		a.tmpDir = os.TempDir()
	}
	_ = os.MkdirAll(a.dataDir, 0o755)
	_ = os.MkdirAll(a.tmpDir, 0o755)

	// one-time: make git reuse gh creds (best-effort)
	_, _ = runCmd("gh", []string{"auth", "setup-git"}, "", 30*time.Second)

	mux := http.NewServeMux()
	mux.HandleFunc("/", a.router)
	srv := &http.Server{Addr: a.addr, Handler: mux, ReadHeaderTimeout: 30 * time.Second}
	logf("data dir = %s", a.dataDir)
	logf("temp dir = %s", a.tmpDir)
	logf("listening on %s (allow=%s)", a.addr, strings.Join(a.allow, ","))
	if err := srv.ListenAndServe(); err != nil {
		logf("server error: %v", err)
		os.Exit(1)
	}
}

// --------------------------------------------------------------------------- //
//
//	routing
//
// --------------------------------------------------------------------------- //
var gitServiceRe = regexp.MustCompile(`^(info/refs|git-upload-pack|git-receive-pack)$`)

func (a *App) router(w http.ResponseWriter, r *http.Request) {
	p := r.URL.Path
	if p == "/" {
		writeJSON(w, 200, infoObj(a))
		return
	}
	if p == "/health" {
		_, err := a.token()
		writeJSON(w, 200, map[string]any{"ok": err == nil, "token": err == nil, "allow": a.allow})
		return
	}
	if p == "/help" || p == "/help/" {
		writeJSON(w, 200, helpOverview(a))
		return
	}
	if strings.HasPrefix(p, "/help/") {
		topic := strings.ToLower(strings.TrimPrefix(p, "/help/"))
		if t, ok := helpTopics[topic]; ok {
			writeJSON(w, 200, t)
			return
		}
		if ps, ok := paramByName[strings.TrimLeft(topic, "-")]; ok {
			writeJSON(w, 200, ps)
			return
		}
		keys := make([]string, 0, len(helpTopics))
		for k := range helpTopics {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		writeJSON(w, 404, map[string]any{"ok": false, "error": "unknown help topic: " + topic, "topics": keys, "params": paramNames()})
		return
	}

	// write: POST /<owner>/<repo>/commits
	if r.Method == http.MethodPost {
		if ow, rp, ok := parseCommitPath(p); ok {
			owner, repo := ow, rp
			if owner == "" {
				// default to first concrete allowed repo
				for _, pat := range a.allow {
					if !strings.Contains(pat, "*") {
						seg := strings.SplitN(pat, "/", 2)
						if len(seg) == 2 {
							owner, repo = seg[0], seg[1]
							break
						}
					}
				}
			}
			if owner == "" {
				writeJSON(w, 400, errObj("specify repo via POST /<owner>/<repo>/commits"))
				return
			}
			a.handleCommit(w, r, owner, repo)
			return
		}
		// write: POST /<owner>/<repo>/branches  （创建分支）
		if ow, rp, ok := parseBranchesPath(p); ok {
			owner, repo := ow, rp
			if owner == "" {
				for _, pat := range a.allow {
					if !strings.Contains(pat, "*") {
						seg := strings.SplitN(pat, "/", 2)
						if len(seg) == 2 {
							owner, repo = seg[0], seg[1]
							break
						}
					}
				}
			}
			if owner == "" {
				writeJSON(w, 400, errObj("specify repo via POST /<owner>/<repo>/branches"))
				return
			}
			a.handleBranchCreate(w, r, owner, repo)
			return
		}
	}

	// read-only metadata endpoints (PRs / CI checks / runs), queried via gh
	if r.Method == http.MethodGet {
		if mo, mr, rest, mok := splitMetaPath(p); mok {
			if !a.allowed(mo, mr) {
				writeJSON(w, 403, errObj("repo not allowed: "+mo+"/"+mr))
				return
			}
			switch {
			case rest == "branches":
				a.handleBranchList(w, r, mo, mr)
				return
			case strings.HasPrefix(rest, "branches/"):
				a.handleBranchView(w, r, mo, mr, strings.TrimPrefix(rest, "branches/"))
				return
			case rest == "pulls":
				a.handlePRList(w, r, mo, mr)
				return
			case strings.HasPrefix(rest, "pulls/") && strings.HasSuffix(rest, "/checks"):
				a.handlePRChecks(w, r, mo, mr, strings.TrimSuffix(strings.TrimPrefix(rest, "pulls/"), "/checks"))
				return
			case strings.HasPrefix(rest, "pulls/"):
				a.handlePRView(w, r, mo, mr, strings.TrimPrefix(rest, "pulls/"))
				return
			case strings.HasPrefix(rest, "commits/") && strings.HasSuffix(rest, "/checks"):
				a.handleChecks(w, r, mo, mr, strings.TrimSuffix(strings.TrimPrefix(rest, "commits/"), "/checks"))
				return
			case rest == "actions/runs":
				a.handleRuns(w, r, mo, mr)
				return
			case rest == "issues":
				a.handleIssueList(w, r, mo, mr)
				return
			case strings.HasPrefix(rest, "issues/"):
				a.handleIssueView(w, r, mo, mr, strings.TrimPrefix(rest, "issues/"))
				return
			}
		}
	}

	// git smart-HTTP: /<owner>/<repo>[.git]/<service>
	owner, repo, service, ok := parseGitPath(p)
	if !ok {
		writeJSON(w, 404, errObj("not found"))
		return
	}
	if !a.allowed(owner, repo) {
		writeJSON(w, 403, errObj("repo not allowed: "+owner+"/"+repo))
		return
	}
	a.proxyGit(w, r, owner, repo, service)
}

func parseGitPath(p string) (owner, repo, service string, ok bool) {
	p = strings.TrimPrefix(p, "/")
	segs := strings.Split(p, "/")
	if len(segs) < 3 {
		return "", "", "", false
	}
	owner = segs[0]
	repo = strings.TrimSuffix(segs[1], ".git")
	rest := strings.Join(segs[2:], "/")
	if !gitServiceRe.MatchString(rest) {
		return "", "", "", false
	}
	return owner, repo, rest, true
}

func parseCommitPath(p string) (owner, repo string, ok bool) {
	if p == "/commits" {
		return "", "", true
	}
	rel := strings.TrimPrefix(p, "/")
	if !strings.HasSuffix(rel, "/commits") {
		return "", "", false
	}
	rel = strings.TrimSuffix(rel, "/commits")
	segs := strings.Split(rel, "/")
	if len(segs) != 2 || segs[0] == "" || segs[1] == "" {
		return "", "", false
	}
	return segs[0], strings.TrimSuffix(segs[1], ".git"), true
}

func (a *App) allowed(owner, repo string) bool {
	full := owner + "/" + repo
	for _, pat := range a.allow {
		if matchRepo(pat, full) {
			return true
		}
	}
	return false
}

func matchRepo(pat, full string) bool {
	if !strings.Contains(pat, "*") {
		return pat == full
	}
	// simple glob: prefix up to *
	pre := strings.SplitN(pat, "*", 2)[0]
	return strings.HasPrefix(full, pre)
}

// --------------------------------------------------------------------------- //
//
//	smart-HTTP reverse proxy
//
// --------------------------------------------------------------------------- //
func (a *App) proxyGit(w http.ResponseWriter, r *http.Request, owner, repo, service string) {
	logf("proxyGit %s %s ?%s (owner=%s repo=%s service=%s)", r.Method, r.URL.Path, r.URL.RawQuery, owner, repo, service)
	target := &url.URL{
		Scheme: "https",
		Host:   "github.com",
		Path:   fmt.Sprintf("/%s/%s.git/%s", owner, repo, service),
	}
	r.URL = &url.URL{Path: target.Path, RawQuery: r.URL.RawQuery}

	// auth
	auth, err := a.authHeader()
	if err != nil {
		writeJSON(w, 500, errObj("auth: "+err.Error()))
		return
	}

	// force-push protection: block non-fast-forward updates to existing refs.
	if service == "git-receive-pack" && r.Method == http.MethodPost && r.Body != nil {
		if handled := a.enforceForce(w, r, owner, repo); handled {
			return
		}
	}

	proxy := httputilReverseProxy(target, auth, owner, repo)
	proxy.ErrorHandler = func(w http.ResponseWriter, req *http.Request, err error) {
		logf("proxy error: %v", err)
		http.Error(w, "proxy error: "+err.Error(), 502)
	}
	proxy.ServeHTTP(w, r)
}

// httputil.ReverseProxy with a Director that injects auth + preserves Git-Protocol.
func httputilReverseProxy(target *url.URL, auth, owner, repo string) *httputil.ReverseProxy {
	p := &httputil.ReverseProxy{}
	p.Director = func(req *http.Request) {
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		req.URL.Path = target.Path
		req.Host = target.Host
		// drop client auth, inject our own
		req.Header.Del("Authorization")
		req.Header.Set("Authorization", auth)
		// preserve Git-Protocol (protocol v2)
		// (Header is already copied from client by ReverseProxy)
		req.Header.Set("User-Agent", "git-proxy")
		// don't let net/http gzip-transport interfere; git handles its own framing
	}
	p.FlushInterval = -1 // stream immediately
	p.Transport = &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		ForceAttemptHTTP2:     false,
		MaxIdleConnsPerHost:   20,
		IdleConnTimeout:       90 * time.Second,
		ResponseHeaderTimeout: 600 * time.Second,
		// disable HTTP/2 to the backend (github.com): h2 flow control has been
		// observed to truncate large smart-HTTP pack streams ("unexpected EOF").
		TLSNextProto: map[string]func(authority string, c *tls.Conn) http.RoundTripper{},
	}
	return p
}

// --------------------------------------------------------------------------- //
//
//	receive-pack force enforcement
//
// --------------------------------------------------------------------------- //
// enforceForce blocks non-fast-forward ("force") updates to refs that already
// exist on the remote. The git smart-HTTP wire carries NO explicit force flag
// (verified against git 2.43), so we detect non-ff by ancestry: for each ref
// update whose old-sha != 0 (existing ref) and new != old, we index the pushed
// pack into the managed clone and run `git merge-base --is-ancestor old new`.
// Fast-forwards, new-branch creates and deletes are allowed; the original body
// is forwarded unchanged. Returns true if a 403 response was written.
// 分支保护策略：gitignore 风格。
//
//	每行：owner/repo/branch-glob        -> 保护
//	      !owner/repo/branch-glob       -> 反选（放行）
//
// 通配：owner/repo 支持 * ?；branch 额外支持 **（跨 /）。
// 语义：按顺序匹配，后匹配的规则覆盖前面的；没有规则匹配 -> 不保护。
// 无任何配置时默认保护每个仓库的 --base 分支；--protect none 关闭。
type branchRule struct {
	ownerRe  *regexp.Regexp
	repoRe   *regexp.Regexp
	branchRe *regexp.Regexp
	negate   bool
}

func (r *branchRule) matches(owner, repo, branch string) bool {
	return r.ownerRe.MatchString(owner) && r.repoRe.MatchString(repo) && r.branchRe.MatchString(branch)
}

type branchPolicy struct {
	rules []branchRule
}

// isProtected：gitignore 语义——顺序匹配，最后一条命中的规则决定结果（! = 不保护）。
func (bp *branchPolicy) isProtected(owner, repo, branch string) bool {
	if bp == nil {
		return false
	}
	protected := false
	for i := range bp.rules {
		if bp.rules[i].matches(owner, repo, branch) {
			protected = !bp.rules[i].negate
		}
	}
	return protected
}

func (a *App) isProtected(owner, repo, refOrBranch string) bool {
	if a.policy == nil {
		return false
	}
	return a.policy.isProtected(owner, repo, strings.TrimPrefix(refOrBranch, "refs/heads/"))
}

// globToRegexp: '*' = 单段 ([^/]*), '**' = 跨 '/' (.*), '?' = 单个非 '/' 字符。
func globToRegexp(pat string) (*regexp.Regexp, error) {
	r := []rune(pat)
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(r); i++ {
		switch r[i] {
		case '*':
			if i+1 < len(r) && r[i+1] == '*' {
				b.WriteString(".*")
				i++
			} else {
				b.WriteString("[^/]*")
			}
		case '?':
			b.WriteString("[^/]")
		default:
			b.WriteString(regexp.QuoteMeta(string(r[i])))
		}
	}
	b.WriteString("$")
	return regexp.Compile(b.String())
}

type policyLine struct {
	text  string
	where string
}

func parseBranchRule(line, where string) (branchRule, error) {
	s := strings.TrimSpace(line)
	neg := false
	if strings.HasPrefix(s, "!") {
		neg = true
		s = strings.TrimSpace(s[1:])
	}
	i1 := strings.IndexByte(s, '/')
	if i1 <= 0 {
		return branchRule{}, fmt.Errorf("%s: bad rule %q: expect owner/repo/branch-glob", where, line)
	}
	rest := s[i1+1:]
	i2 := strings.IndexByte(rest, '/')
	if i2 <= 0 {
		return branchRule{}, fmt.Errorf("%s: bad rule %q: expect owner/repo/branch-glob", where, line)
	}
	owner, repo, branch := s[:i1], rest[:i2], rest[i2+1:]
	if branch == "" {
		return branchRule{}, fmt.Errorf("%s: bad rule %q: empty branch glob", where, line)
	}
	ownerRe, err := globToRegexp(owner)
	if err != nil {
		return branchRule{}, fmt.Errorf("%s: bad owner %q: %v", where, owner, err)
	}
	repoRe, err := globToRegexp(repo)
	if err != nil {
		return branchRule{}, fmt.Errorf("%s: bad repo %q: %v", where, repo, err)
	}
	branchRe, err := globToRegexp(branch)
	if err != nil {
		return branchRule{}, fmt.Errorf("%s: bad branch %q: %v", where, branch, err)
	}
	return branchRule{ownerRe: ownerRe, repoRe: repoRe, branchRe: branchRe, negate: neg}, nil
}

// parsePolicyFile 读取 gitignore 风格策略文件：一行一条，'#' 注释、空行忽略。
func parsePolicyFile(path string) ([]policyLine, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []policyLine
	for i, line := range strings.Split(string(b), "\n") {
		if j := strings.IndexByte(line, '#'); j >= 0 {
			line = line[:j]
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		out = append(out, policyLine{text: line, where: fmt.Sprintf("%s:%d", path, i+1)})
	}
	return out, nil
}

// loadBranchPolicy 构建策略：文件行在前，内联 --protect 行在后（所以内联可覆盖文件）。
func loadBranchPolicy(files, inline []string, base string) (*branchPolicy, error) {
	for _, v := range inline {
		if strings.EqualFold(strings.TrimSpace(v), "none") {
			return &branchPolicy{}, nil
		}
	}
	var lines []policyLine
	for _, f := range files {
		ls, err := parsePolicyFile(f)
		if err != nil {
			return nil, err
		}
		lines = append(lines, ls...)
	}
	for i, v := range inline {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		lines = append(lines, policyLine{text: v, where: fmt.Sprintf("--protect[%d]", i)})
	}
	if len(lines) == 0 {
		lines = []policyLine{{text: "*/*/" + base, where: "default"}}
	}
	bp := &branchPolicy{}
	for _, ln := range lines {
		r, err := parseBranchRule(ln.text, ln.where)
		if err != nil {
			return nil, err
		}
		bp.rules = append(bp.rules, r)
	}
	return bp, nil
}

func (a *App) enforceForce(w http.ResponseWriter, r *http.Request, owner, repo string) bool {
	br := bufio.NewReader(r.Body)
	peek, _ := br.Peek(1 << 20) // 1MiB: enough for the command list on normal pushes
	cmds, flushOff, ok := parseReceiveCmds(peek)
	logf("enforceForce: cmds=%d flushOff=%d ok=%v", len(cmds), flushOff, ok)
	if !ok {
		// command list not fully within peek (huge push) or unparseable -> forward untouched
		r.Body = io.NopCloser(br)
		return false
	}
	// deleting branches/tags is NOT allowed
	for _, c := range cmds {
		if c.new == zeroSHA {
			logf("BLOCK delete of %s", c.ref)
			writeJSON(w, 403, map[string]any{"ok": false, "error": "deleting branches/tags is forbidden: " + c.ref})
			return true
		}
	}
	// pushes to protected branches are NOT allowed (no anonymous merge into main)
	for _, c := range cmds {
		if a.isProtected(owner, repo, c.ref) {
			logf("BLOCK push to protected branch %s", c.ref)
			writeJSON(w, 403, map[string]any{"ok": false, "error": "push to protected branch is forbidden: " + c.ref + " (open a PR instead)"})
			return true
		}
	}
	var updates []refCmd
	for _, c := range cmds {
		if c.old != zeroSHA && c.new != zeroSHA && c.old != c.new {
			updates = append(updates, c)
		}
	}
	logf("enforceForce: cmds=%d updates-to-existing=%d", len(cmds), len(updates))
	if len(updates) == 0 {
		// only creates -> forward untouched
		r.Body = io.NopCloser(br)
		return false
	}
	// buffer the full body to extract the pack and check ancestry (cap 256MiB)
	full, err := io.ReadAll(io.LimitReader(br, 256<<20))
	if err != nil {
		r.Body = io.NopCloser(bytes.NewReader(full))
		r.ContentLength = int64(len(full))
		return false
	}
	if int64(len(full)) >= 256<<20 {
		// hit the cap -> skip enforcement, forward what we have
		r.Body = io.NopCloser(bytes.NewReader(full))
		r.ContentLength = int64(len(full))
		return false
	}
	packOff := flushOff + 4 // bytes after the flush pkt '0000'
	var pack []byte
	if packOff < len(full) {
		pack = full[packOff:]
	}
	blockedRef := ""
	func() {
		a.cloneMu.Lock()
		defer a.cloneMu.Unlock()
		dir := filepath.Join(a.dataDir, owner+"-"+repo)
		if err := a.ensureClone(owner, repo, dir); err != nil {
			logf("force-check: ensureClone failed: %v (skipping enforcement)", err)
			return
		}
		env := gitEnv("")
		// short-timeout, best-effort: never hang a push on the check
		runCmd("git", []string{"fetch", "origin", "--quiet"}, dir, 30*time.Second, env...)
		if len(pack) > 0 {
			if code, out := runCmdStdin("git", []string{"index-pack", "--stdin", "--fix-thin"}, dir, pack, 30*time.Second, env...); code != 0 {
				logf("force-check: index-pack failed (skipping enforcement): %s", strings.TrimSpace(out))
				return
			}
		}
		for _, c := range updates {
			code, _ := runCmdCode("git", []string{"merge-base", "--is-ancestor", c.old, c.new}, dir, 10*time.Second, env...)
			if code != 0 {
				blockedRef = c.ref
				return
			}
		}
	}()
	if blockedRef != "" {
		logf("BLOCK force-push (non-fast-forward) to %s", blockedRef)
		writeJSON(w, 403, map[string]any{"ok": false, "error": "force-push (non-fast-forward) to existing ref forbidden: " + blockedRef})
		return true
	}
	// forward the original body unchanged
	r.Body = io.NopCloser(bytes.NewReader(full))
	r.ContentLength = int64(len(full))
	return false
}

type refCmd struct{ old, new, ref string }

// parseReceiveCmds parses pkt-line ref-update commands from the start of a
// git-receive-pack body. Returns commands (excluding shallow/etc), the byte
// offset of the flush pkt ending the command list, and ok=true if the flush
// was found within the buffer.
func parseReceiveCmds(buf []byte) (cmds []refCmd, flushOff int, ok bool) {
	off := 0
	for off+4 <= len(buf) {
		length, err := parseHexLen(buf[off : off+4])
		if err != nil {
			return cmds, 0, false
		}
		if length == 0 {
			return cmds, off, true // flush pkt
		}
		if length < 4 || off+length > len(buf) {
			return cmds, 0, false // not fully in buffer
		}
		payload := buf[off+4 : off+length]
		off += length
		if i := bytes.IndexByte(payload, 0); i >= 0 { // strip capabilities (NUL)
			payload = payload[:i]
		}
		s := strings.TrimSpace(string(payload))
		if s == "" || strings.HasPrefix(s, "shallow") || strings.HasPrefix(s, "unshallow") {
			continue
		}
		p := strings.Fields(strings.TrimPrefix(s, "+"))
		if len(p) >= 3 && shaRe.MatchString(p[0]) && shaRe.MatchString(p[1]) {
			cmds = append(cmds, refCmd{old: p[0], new: p[1], ref: p[2]})
		}
	}
	return cmds, 0, false
}

func parseHexLen(b []byte) (int, error) {
	if len(b) != 4 {
		return 0, errors.New("bad len")
	}
	var n int
	for _, c := range b {
		var d int
		switch {
		case c >= '0' && c <= '9':
			d = int(c - '0')
		case c >= 'a' && c <= 'f':
			d = int(c-'a') + 10
		case c >= 'A' && c <= 'F':
			d = int(c-'A') + 10
		default:
			return 0, errors.New("bad hex")
		}
		n = n*16 + d
	}
	if n != 0 && n < 4 {
		return 0, errors.New("len too small")
	}
	return n, nil
}

func pktLine(payload []byte) []byte {
	return []byte(fmt.Sprintf("%04x", 4+len(payload)) + string(payload))
}

// --------------------------------------------------------------------------- //
//
//	POST /commits — apply patch, commit, push, optionally PR
//
// --------------------------------------------------------------------------- //
type commitResult struct {
	OK        bool   `json:"ok"`
	Status    string `json:"status"`
	Owner     string `json:"owner,omitempty"`
	Repo      string `json:"repo,omitempty"`
	Branch    string `json:"branch"`
	Sha       string `json:"sha,omitempty"`
	Short     string `json:"short,omitempty"`
	Message   string `json:"message,omitempty"`
	Error     string `json:"error,omitempty"`
	Detail    string `json:"detail,omitempty"`
	PR        string `json:"pr,omitempty"`
	ElapsedMs int64  `json:"elapsedMs,omitempty"`
}

func (a *App) handleCommit(w http.ResponseWriter, r *http.Request, owner, repo string) {
	if !a.allowed(owner, repo) {
		writeJSON(w, 403, errObj("repo not allowed: "+owner+"/"+repo))
		return
	}
	if a.apiToken != "" {
		auth := r.Header.Get("Authorization")
		if !(strings.HasPrefix(auth, "Bearer ") && auth[7:] == a.apiToken) && r.URL.Query().Get("token") != a.apiToken {
			writeJSON(w, 401, errObj("unauthorized"))
			return
		}
	}

	var branch, message, base, prSpec string
	var patch []byte
	ctype := r.Header.Get("Content-Type")

	if strings.Contains(ctype, "application/json") {
		var j struct {
			Branch  string `json:"branch"`
			Message string `json:"message"`
			Patch   string `json:"patch"`
			Base    string `json:"base"`
			PR      string `json:"pr"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 64<<20)).Decode(&j); err != nil {
			writeJSON(w, 400, errObj("invalid JSON: "+err.Error()))
			return
		}
		branch, message, base, prSpec, patch = j.Branch, j.Message, j.Base, j.PR, []byte(j.Patch)
	} else {
		q := r.URL.Query()
		branch = q.Get("branch")
		message = q.Get("message")
		if message == "" {
			message = r.Header.Get("X-Message")
		}
		base = q.Get("base")
		prSpec = q.Get("pr")
		if prSpec == "" {
			prSpec = q.Get("pr_title")
		}
		patch, _ = io.ReadAll(io.LimitReader(r.Body, 64<<20))
	}
	if branch == "" {
		writeJSON(w, 400, errObj("missing 'branch'"))
		return
	}
	if message == "" && prSpec == "" {
		writeJSON(w, 400, errObj("missing 'message'"))
		return
	}
	if len(patch) == 0 {
		writeJSON(w, 400, errObj("empty patch"))
		return
	}
	if base == "" {
		base = a.base
	}

	t0 := time.Now()
	res := a.applyPatchCommit(owner, repo, branch, message, base, prSpec, patch)
	res.ElapsedMs = time.Since(t0).Milliseconds()
	res.Owner, res.Repo = owner, repo
	logf("commit %s/%s:%s -> %s %s (%dms)", owner, repo, branch, res.Status, res.Short, res.ElapsedMs)
	code := 200
	if !res.OK {
		switch {
		case res.Error == "protected branch":
			code = 403
		case strings.Contains(res.Error, "push"):
			code = 502
		default:
			code = 400
		}
	}
	writeJSON(w, code, res)
}

func (a *App) applyPatchCommit(owner, repo, branch, message, base, prSpec string, patch []byte) commitResult {
	if a.isProtected(owner, repo, branch) {
		return commitResult{OK: false, Status: "failed", Error: "protected branch", Detail: "commit/push to protected branch '" + branch + "' is not allowed; use a feature branch and open a PR", Branch: branch}
	}
	a.cloneMu.Lock()
	defer a.cloneMu.Unlock()

	dir := filepath.Join(a.dataDir, owner+"-"+repo)
	if err := a.ensureClone(owner, repo, dir); err != nil {
		return commitResult{OK: false, Status: "failed", Error: "clone", Detail: err.Error(), Branch: branch}
	}

	auth, _ := a.authHeader()
	env := gitEnv(auth)

	// fetch
	if out, err := gitRun(dir, env, "fetch", "origin", "--prune", "--quiet"); err != nil {
		return commitResult{OK: false, Status: "failed", Error: "git error", Detail: out + err.Error(), Branch: branch}
	}
	// resolve start point: existing remote <branch>, else base = branch name OR commit sha
	exists := refExists(dir, env, "refs/remotes/origin/"+branch)
	startRef := "origin/" + branch
	if !exists {
		startRef = "origin/" + base
		if !refExists(dir, env, startRef) && shaLikeRe.MatchString(base) && refExists(dir, env, base) {
			startRef = base // base given as a commit sha (reachable from a fetched ref)
		}
	}
	if !refExists(dir, env, startRef) {
		startRef = "HEAD"
	}
	tip, err := gitRun(dir, env, "rev-parse", startRef)
	if err != nil {
		return commitResult{OK: false, Status: "failed", Error: "git error", Detail: "rev-parse: " + tip, Branch: branch}
	}
	tip = strings.TrimSpace(tip)

	// clean + switch
	_, _ = gitRun(dir, env, "reset", "--hard", "HEAD", "--quiet")
	_, _ = gitRun(dir, env, "clean", "-fdq")
	if out, err := gitRun(dir, env, "checkout", "-B", branch, tip); err != nil {
		return commitResult{OK: false, Status: "failed", Error: "git error", Detail: out + err.Error(), Branch: branch}
	}
	_, _ = gitRun(dir, env, "clean", "-fdq")

	// apply
	if applied, detail := a.applyPatchFile(dir, env, patch); !applied {
		return commitResult{OK: false, Status: "failed", Error: "patch does not apply cleanly", Detail: detail, Branch: branch, Short: tip}
	}
	// stage + commit
	_, _ = gitRun(dir, env, "add", "-A")
	if code, _ := gitRunCode(dir, env, "diff", "--cached", "--quiet"); code == 0 {
		sha, _ := gitRun(dir, env, "rev-parse", "HEAD")
		return commitResult{OK: true, Status: "noop", Branch: branch, Short: strings.TrimSpace(sha), Message: message}
	}
	if message == "" {
		message = "apply patch"
	}
	author := os.Getenv("GIT_BRIDGE_AUTHOR")
	if author == "" {
		// default from gh user
		if u, _ := runCmd("gh", []string{"api", "user", "--jq", ".login"}, "", 15*time.Second); strings.TrimSpace(u) != "" {
			author = strings.TrimSpace(u)
		}
	}
	commitArgs := []string{"commit", "--quiet", "-m", message}
	if author != "" {
		commitArgs = append(commitArgs, "--author", author+" <"+author+"@users.noreply.github.com>")
	}
	if out, err := gitRun(dir, env, commitArgs...); err != nil {
		return commitResult{OK: false, Status: "failed", Error: "git error", Detail: out + err.Error(), Branch: branch}
	}
	sha, _ := gitRun(dir, env, "rev-parse", "HEAD")
	short, _ := gitRun(dir, env, "rev-parse", "--short=7", "HEAD")
	sha = strings.TrimSpace(sha)
	short = strings.TrimSpace(short)

	// push
	out, err := gitRun(dir, env, "push", "origin", "HEAD:refs/heads/"+branch)
	if err != nil {
		return commitResult{OK: false, Status: "failed", Error: "push failed", Detail: strings.TrimSpace(out), Branch: branch, Sha: sha, Short: short}
	}

	res := commitResult{OK: true, Status: "pushed", Branch: branch, Sha: sha, Short: short, Message: message}

	// optional PR
	if prSpec != "" {
		title, body := splitPR(prSpec, message)
		prURL, perr := createPR(owner, repo, base, branch, title, body)
		if perr != nil {
			res.PR = "error: " + perr.Error()
		} else {
			res.PR = prURL
		}
	}
	return res
}

func splitPR(spec, fallbackMsg string) (title, body string) {
	if spec == "1" || spec == "true" {
		return firstLine(fallbackMsg), ""
	}
	if i := strings.Index(spec, "::"); i >= 0 {
		return spec[:i], spec[i+2:]
	}
	return spec, ""
}
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

func (a *App) ensureClone(owner, repo, dir string) error {
	if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
		return nil
	}
	_ = os.RemoveAll(dir)
	auth, err := a.authHeader()
	if err != nil {
		return err
	}
	env := gitEnv(auth)
	// retry: SSH clone of larger repos can hit transient "early EOF" on flaky links
	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		out, err := runCmd("git", []string{"clone", "--", "git@github.com:" + owner + "/" + repo + ".git", dir}, "", 180*time.Second, env...)
		if err == nil {
			return nil
		}
		lastErr = fmt.Errorf("clone attempt %d: %s", attempt, strings.TrimSpace(out))
		logf("ensureClone: %v", lastErr)
		_ = os.RemoveAll(dir)
		time.Sleep(2 * time.Second)
	}
	return lastErr
}

func createPR(owner, repo, base, head, title, body string) (string, error) {
	args := []string{"pr", "create", "--repo", owner + "/" + repo, "--base", base, "--head", head, "--title", title, "--body", body}
	out, err := runCmd("gh", args, "", 60*time.Second)
	if err != nil {
		return "", fmt.Errorf("%s", strings.TrimSpace(out+err.Error()))
	}
	return strings.TrimSpace(out), nil
}

// --------------------------------------------------------------------------- //
//
//	git helpers
//
// --------------------------------------------------------------------------- //
func gitEnv(authHeader string) []string {
	e := append([]string{}, os.Environ()...)
	e = append(e, "GIT_TERMINAL_PROMPT=0", "GIT_ASKPASS=echo")
	if authHeader != "" {
		e = append(e, "GIT_HTTP_AUTHORIZATION="+authHeader)
	}
	return e
}

func gitRun(dir string, env []string, args ...string) (string, error) {
	out, err := runCmd("git", args, dir, 120*time.Second, env...)
	return out, err
}
func gitRunCode(dir string, env []string, args ...string) (int, string) {
	return runCmdCode("git", args, dir, 120*time.Second, env...)
}
func runCmdStdin(name string, args []string, dir string, stdin []byte, timeout time.Duration, env ...string) (int, string) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	c := exec.CommandContext(ctx, name, args...)
	c.Dir = dir
	c.Stdin = bytes.NewReader(stdin)
	if len(env) > 0 {
		c.Env = env
	}
	var bout bytes.Buffer
	c.Stdout = &bout
	c.Stderr = &bout
	if err := c.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return -1, bout.String() + "\n(timeout)"
		}
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), bout.String()
		}
		return -1, bout.String() + "\n" + err.Error()
	}
	return 0, bout.String()
}

func refExists(dir string, env []string, ref string) bool {
	code, _ := gitRunCode(dir, env, "rev-parse", "--verify", "--quiet", ref+"^{commit}")
	return code == 0
}

func (a *App) applyPatchFile(dir string, env []string, patch []byte) (bool, string) {
	tmp := filepath.Join(a.tmpDir, "gp-"+randHex(6)+".patch")
	if err := os.WriteFile(tmp, patch, 0o644); err != nil {
		return false, err.Error()
	}
	defer os.Remove(tmp)
	code, out := gitRunCode(dir, env, "apply", "--whitespace=nowarn", "--check", tmp)
	if code != 0 {
		return false, strings.TrimSpace(out)
	}
	_, _ = gitRun(dir, env, "apply", "--whitespace=nowarn", tmp)
	return true, ""
}

// --------------------------------------------------------------------------- //
//
//	command runner
//
// --------------------------------------------------------------------------- //
func runCmd(name string, args []string, dir string, timeout time.Duration, env ...string) (string, error) {
	code, out := runCmdCode(name, args, dir, timeout, env...)
	if code != 0 {
		return out, fmt.Errorf("exit %d", code)
	}
	return out, nil
}
func runCmdCode(name string, args []string, dir string, timeout time.Duration, env ...string) (int, string) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	c := exec.CommandContext(ctx, name, args...)
	c.Dir = dir
	c.Stdin = nil
	if len(env) > 0 {
		c.Env = env
	}
	var bout bytes.Buffer
	c.Stdout = &bout
	c.Stderr = &bout
	if err := c.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return -1, bout.String() + "\n(timeout)"
		}
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), bout.String()
		}
		return -1, bout.String() + "\n" + err.Error()
	}
	return 0, bout.String()
}

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// --------------------------------------------------------------------------- //
//
//	token
//
// --------------------------------------------------------------------------- //
func (a *App) token() (string, error) {
	a.tokenMu.Lock()
	defer a.tokenMu.Unlock()
	if a.tokCache != "" && time.Since(a.tokCacheAt) < 5*time.Minute {
		return a.tokCache, nil
	}
	out, err := runCmd("gh", []string{"auth", "token"}, "", 15*time.Second)
	if err != nil {
		return "", fmt.Errorf("gh auth token: %s", strings.TrimSpace(out))
	}
	a.tokCache = strings.TrimSpace(out)
	a.tokCacheAt = time.Now()
	return a.tokCache, nil
}
func (a *App) authHeader() (string, error) {
	tok, err := a.token()
	if err != nil {
		return "", err
	}
	return "Basic " + base64.StdEncoding.EncodeToString([]byte("x-access-token:"+tok)), nil
}

// --------------------------------------------------------------------------- //
//
//	http helpers
//
// --------------------------------------------------------------------------- //
func writeJSON(w http.ResponseWriter, code int, v any) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false) // keep <, >, & readable in API output
	if err := enc.Encode(v); err != nil {
		buf.Reset()
		buf.WriteString(`{"ok":false,"error":"json encode failed"}`)
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	w.Write(buf.Bytes())
}
func errObj(msg string) map[string]any { return map[string]any{"ok": false, "error": msg} }
func infoObj(a *App) map[string]any {
	return map[string]any{
		"service": "gitproxy",
		"usage":   "git clone https://<host>/<owner>/<repo>   |   POST /<owner>/<repo>/commits?branch=&message=&pr=",
		"endpoints": map[string]any{
			"POST /<owner>/<repo>/commits":                             "提交 patch(diff) → commit → push；?branch=&message=&base=&pr=",
			"GET /<owner>/<repo>/branches":                             "分支列表；?limit=N",
			"GET /<owner>/<repo>/branches/<name>":                      "分支详情（指向的 commit）",
			"POST /<owner>/<repo>/branches":                            "创建分支；body/query: name=&from=（分支/commit sha/tag，默认 --base）",
			"GET /<owner>/<repo>/pulls":                                "PR 列表；?state=open|closed|merged|all&limit=N",
			"GET /<owner>/<repo>/pulls/<n>":                            "PR 详情 + statusCheckRollup(CI)",
			"GET /<owner>/<repo>/pulls/<n>/checks":                     "PR 头提交的 CI check-runs",
			"GET /<owner>/<repo>/commits/<sha>/checks":                 "某 commit 的 check-runs + combined status",
			"GET /<owner>/<repo>/actions/runs?branch=&limit=N":         "最近 GitHub Actions 运行",
			"GET /<owner>/<repo>/issues?state=open|closed|all&limit=N": "issue 列表",
			"GET /<owner>/<repo>/issues/<n>":                           "issue 详情 + comments",
			"GET /health":                                              "健康检查",
			"GET /help[/<topic>]":                                      "总览 / 分主题帮助（clone/push/commits/pulls/issues/checks/runs/force/delete/auth/whitelist/errors）",
		},
		"allow": a.allow,
	}
}
func logf(f string, args ...any) {
	fmt.Fprintf(os.Stdout, "[gitproxy] "+f+"\n", args...)
}

// readAllowFile reads allowed repos from a text file, one per line.
// blank lines and lines beginning with '#' are ignored.
func readAllowFile(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, line := range strings.Split(string(b), "\n") {
		if i := strings.IndexByte(line, '#'); i >= 0 {
			line = line[:i]
		}
		line = strings.TrimSpace(line)
		if line != "" {
			out = append(out, line)
		}
	}
	return out, nil
}

func dedupeStrings(in []string) []string {
	seen := make(map[string]bool, len(in))
	out := make([]string, 0, len(in))
	for _, v := range in {
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	return out
}

type arrayFlag []string

func (a *arrayFlag) String() string { return strings.Join(*a, ",") }
func (a *arrayFlag) Set(v string) error {
	for _, x := range strings.Split(v, ",") {
		if x = strings.TrimSpace(x); x != "" {
			*a = append(*a, x)
		}
	}
	return nil
}

// --------------------------------------------------------------------------- //
//
//	read-only metadata endpoints (PR / CI) — proxied through gh, server-side token
//
// --------------------------------------------------------------------------- //
func splitMetaPath(p string) (owner, repo, rest string, ok bool) {
	p = strings.TrimPrefix(p, "/")
	if p == "" {
		return "", "", "", false
	}
	segs := strings.SplitN(p, "/", 3)
	if len(segs) < 2 || segs[0] == "" || segs[1] == "" {
		return "", "", "", false
	}
	owner = segs[0]
	repo = strings.TrimSuffix(segs[1], ".git")
	if len(segs) == 3 {
		rest = segs[2]
	}
	return owner, repo, rest, true
}

func clampInt(s string, lo, hi, def int) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	if n < lo {
		n = lo
	}
	if n > hi {
		n = hi
	}
	return n
}

func writeRawJSON(w http.ResponseWriter, body string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(200)
	io.WriteString(w, body)
}

func ghOut(args ...string) (string, error) {
	out, err := runCmd("gh", args, "", 30*time.Second)
	if err != nil {
		return out, fmt.Errorf("%s", strings.TrimSpace(out))
	}
	return out, nil
}

func (a *App) handlePRList(w http.ResponseWriter, r *http.Request, owner, repo string) {
	q := r.URL.Query()
	limit := clampInt(q.Get("limit"), 1, 100, 20)
	args := []string{"pr", "list", "--repo", owner + "/" + repo,
		"--json", "number,title,state,headRefName,baseRefName,isDraft,mergeable,url,updatedAt,author",
		"--limit", strconv.Itoa(limit)}
	if st := q.Get("state"); st != "" {
		args = append(args, "--state", st)
	}
	out, err := ghOut(args...)
	if err != nil {
		writeJSON(w, 502, errObj("gh pr list: "+err.Error()))
		return
	}
	writeRawJSON(w, `{"pulls":`+strings.TrimSpace(out)+`}`)
}

func (a *App) handlePRView(w http.ResponseWriter, r *http.Request, owner, repo, num string) {
	args := []string{"pr", "view", num, "--repo", owner + "/" + repo,
		"--json", "number,title,state,headRefName,baseRefName,isDraft,mergeable,mergeStateStatus,url,reviewDecision,statusCheckRollup,commits,author"}
	out, err := ghOut(args...)
	if err != nil {
		writeJSON(w, 502, errObj("gh pr view: "+err.Error()))
		return
	}
	writeRawJSON(w, `{"pull":`+strings.TrimSpace(out)+`}`)
}

func (a *App) handleChecks(w http.ResponseWriter, r *http.Request, owner, repo, sha string) {
	if !shaLikeRe.MatchString(sha) {
		writeJSON(w, 400, errObj("bad sha (expect 7-40 hex)"))
		return
	}
	base := "repos/" + owner + "/" + repo + "/commits/" + sha
	out, err := ghOut("api", base+"/check-runs")
	if err != nil {
		writeJSON(w, 502, errObj("gh api check-runs: "+err.Error()))
		return
	}
	st := "null"
	if s, e := ghOut("api", base+"/status"); e == nil && strings.TrimSpace(s) != "" {
		st = strings.TrimSpace(s)
	}
	writeRawJSON(w, `{"check_runs":`+strings.TrimSpace(out)+`,"combined_status":`+st+`}`)
}

func (a *App) handleRuns(w http.ResponseWriter, r *http.Request, owner, repo string) {
	q := r.URL.Query()
	limit := clampInt(q.Get("limit"), 1, 100, 10)
	args := []string{"run", "list", "--repo", owner + "/" + repo,
		"--json", "databaseId,name,displayTitle,status,conclusion,headBranch,headSha,url,createdAt",
		"--limit", strconv.Itoa(limit)}
	if br := q.Get("branch"); br != "" {
		args = append(args, "--branch", br)
	}
	out, err := ghOut(args...)
	if err != nil {
		writeJSON(w, 502, errObj("gh run list: "+err.Error()))
		return
	}
	writeRawJSON(w, `{"runs":`+strings.TrimSpace(out)+`}`)
}

func jsonStr(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func (a *App) handleIssueList(w http.ResponseWriter, r *http.Request, owner, repo string) {
	q := r.URL.Query()
	limit := clampInt(q.Get("limit"), 1, 100, 20)
	args := []string{"issue", "list", "--repo", owner + "/" + repo,
		"--json", "number,title,state,labels,author,url,updatedAt,comments",
		"--limit", strconv.Itoa(limit)}
	if st := q.Get("state"); st != "" {
		args = append(args, "--state", st)
	}
	out, err := ghOut(args...)
	if err != nil {
		writeJSON(w, 502, errObj("gh issue list: "+err.Error()))
		return
	}
	writeRawJSON(w, `{"issues":`+strings.TrimSpace(out)+`}`)
}

func (a *App) handleIssueView(w http.ResponseWriter, r *http.Request, owner, repo, num string) {
	if _, err := strconv.Atoi(num); err != nil {
		writeJSON(w, 400, errObj("bad issue number"))
		return
	}
	args := []string{"issue", "view", num, "--repo", owner + "/" + repo,
		"--json", "number,title,state,body,labels,author,url,updatedAt,comments,assignees,createdAt"}
	out, err := ghOut(args...)
	if err != nil {
		writeJSON(w, 502, errObj("gh issue view: "+err.Error()))
		return
	}
	writeRawJSON(w, `{"issue":`+strings.TrimSpace(out)+`}`)
}

func (a *App) handlePRChecks(w http.ResponseWriter, r *http.Request, owner, repo, num string) {
	if _, err := strconv.Atoi(num); err != nil {
		writeJSON(w, 400, errObj("bad PR number"))
		return
	}
	out, err := ghOut("pr", "view", num, "--repo", owner+"/"+repo, "--json", "number,title,state,headRefOid")
	if err != nil {
		writeJSON(w, 502, errObj("gh pr view: "+err.Error()))
		return
	}
	var pv struct {
		Number     int    `json:"number"`
		Title      string `json:"title"`
		State      string `json:"state"`
		HeadRefOid string `json:"headRefOid"`
	}
	if err := json.Unmarshal([]byte(out), &pv); err != nil || pv.HeadRefOid == "" {
		writeJSON(w, 502, errObj("cannot resolve PR head sha"))
		return
	}
	base := "repos/" + owner + "/" + repo + "/commits/" + pv.HeadRefOid
	cr, e := ghOut("api", base+"/check-runs")
	if e != nil {
		writeJSON(w, 502, errObj("gh api check-runs: "+e.Error()))
		return
	}
	st := "null"
	if sv, e2 := ghOut("api", base+"/status"); e2 == nil && strings.TrimSpace(sv) != "" {
		st = strings.TrimSpace(sv)
	}
	writeRawJSON(w, `{"pull":`+num+`,"title":`+jsonStr(pv.Title)+`,"state":`+jsonStr(pv.State)+`,"headSha":`+jsonStr(pv.HeadRefOid)+`,"check_runs":`+strings.TrimSpace(cr)+`,"combined_status":`+st+`}`)
}

// --------------------------------------------------------------------------- //
//
//	/help
//
// --------------------------------------------------------------------------- //
type helpTopic struct {
	Topic    string   `json:"topic"`
	Title    string   `json:"title"`
	Summary  string   `json:"summary"`
	Usage    []string `json:"usage,omitempty"`
	Examples []string `json:"examples,omitempty"`
	Notes    []string `json:"notes,omitempty"`
	Related  []string `json:"related,omitempty"`
}

var helpTopics = map[string]helpTopic{
	"clone": {
		Topic:   "clone",
		Title:   "clone — 经代理拉取仓库",
		Summary: "用标准 git clone 指向中转 URL 即可；代理把 smart-HTTP 转发到 GitHub 并注入鉴权，agent 无需任何凭据。",
		Usage:   []string{"git clone <BASE>/<owner>/<repo>", "git clone --depth 1 --branch <branch> <BASE>/<owner>/<repo>"},
		Examples: []string{
			"git clone https://wsl-8000.moonchan.xyz/Hana-ame/agent",
			"git clone --depth 1 --branch main https://wsl-8000.moonchan.xyz/Hana-ame/agent",
			"git clone --shallow-since=2026-10-01 --branch main https://wsl-8000.moonchan.xyz/Hana-ame/agent",
		},
		Notes: []string{
			"仓库必须在白名单内，否则 403。",
			"支持 --branch / --depth N / --shallow-since / --unshallow / --deepen。",
			"偶发 invalid index-pack output 时重试即可（后端已强制 HTTP/1.1 降低概率）。",
		},
		Related: []string{"push", "pull", "whitelist"},
	},
	"push": {
		Topic:   "push",
		Title:   "push — 经 HTTPS 中转推送",
		Summary: "普通 git push 指向中转 URL；代理转发 git-receive-pack 到 GitHub（注入 token）。",
		Usage:   []string{"git push origin <branch>"},
		Examples: []string{
			"git push origin feat-x",
			"git push -u origin feat-x",
		},
		Notes: []string{
			"新建分支 / fast-forward 允许。",
			"non-fast-forward 改写已有分支（-f / --force-with-lease）→ 403。",
			"删除分支 / tag（git push --delete / :refs/tags/x）→ 403。",
		},
		Related: []string{"push", "force", "delete"},
	},
	"pull": {
		Topic:    "pull",
		Title:    "pull — fetch + merge",
		Summary:  "git pull 经代理执行，等价于 fetch + merge，走 upload-pack。",
		Usage:    []string{"git pull origin <branch>", "git pull"},
		Examples: []string{"git pull origin main", "git fetch origin --depth 1"},
		Notes:    []string{"合并冲突是纯客户端行为，代理只转发数据。"},
		Related:  []string{"clone", "merge"},
	},
	"merge": {
		Topic:   "merge",
		Title:   "merge — 本地合并后推送",
		Summary: "在本地 git merge 生成合并提交，再经代理 push，与普通 push 一致。",
		Usage:   []string{"git checkout <base>", "git merge --no-ff <feat>", "git push origin <base>"},
		Examples: []string{
			"git checkout base-branch",
			"git merge --no-ff feat-branch -m 'merge feat into base'",
			"git push origin base-branch",
		},
		Notes:   []string{"合并提交是 fast-forward 到已有分支的普通 push，放行。"},
		Related: []string{"push", "pull"},
	},
	"commits": {
		Topic:   "commits",
		Title:   "commits — 提交 patch（apply → commit → push）",
		Summary: "把 unified diff POST 过来，服务端在托管 clone 里 fetch→apply→commit→push；可选自动开 PR。",
		Usage: []string{
			"POST <BASE>/<owner>/<repo>/commits?branch=<b>&message=<m>[&base=<b|sha>][&pr=1|<title>::<body>]",
			"body 为 raw unified diff（Content-Type 任意），或 JSON {branch,message,patch,base,pr}",
		},
		Examples: []string{
			"git diff > my.patch",
			"curl -X POST 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/commits?branch=feat-x&message=msg&pr=1' --data-binary @my.patch",
			`curl -X POST 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/commits' -H 'Content-Type: application/json' -d '{"branch":"feat","message":"msg","patch":"--- /dev/null\n+++ b/f\n@@ -0,0 +1 @@\n+hi\n"}'`,
		},
		Notes: []string{
			"branch 不存在会从 base（默认 main）新建；base 可传分支名或 commit sha。",
			"pr=1 标题取自 message；pr=<title>::<body> 可自定义 PR 标题/正文。",
			"patch 必须是 git diff 的 unified diff，不要发 format-patch 的 mbox。",
			"错误：缺 branch/message/空 patch/patch 应用失败 → 400；push 失败 → 502。",
			"设了 --api-token 时需 Authorization: Bearer <token>（或 ?token=）。",
		},
		Related: []string{"pulls", "checks"},
	},
	"pulls": {
		Topic:   "pulls",
		Title:   "pulls — PR 列表 / 详情 / CI",
		Summary: "只读查询 GitHub PR（服务端 gh 拉取，agent 无需凭据）。",
		Usage: []string{
			"GET <BASE>/<owner>/<repo>/pulls?state=open|closed|merged|all&limit=N",
			"GET <BASE>/<owner>/<repo>/pulls/<n>",
			"GET <BASE>/<owner>/<repo>/pulls/<n>/checks",
		},
		Examples: []string{
			"curl -sS 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/pulls?state=open'",
			"curl -sS 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/pulls/7'",
			"curl -sS 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/pulls/7/checks'",
		},
		Notes: []string{
			"响应 key：列表 {pulls:[...]}，详情 {pull:{...,statusCheckRollup:[...]}}。",
			"坏编号 → 400；不存在的 PR → 502。",
		},
		Related: []string{"commits", "checks"},
	},
	"issues": {
		Topic:   "issues",
		Title:   "issues — issue 列表 / 详情",
		Summary: "只读查询 GitHub issue 与评论。",
		Usage: []string{
			"GET <BASE>/<owner>/<repo>/issues?state=open|closed|all&limit=N",
			"GET <BASE>/<owner>/<repo>/issues/<n>",
		},
		Examples: []string{
			"curl -sS 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/issues?state=all'",
			"curl -sS 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/issues/1'",
		},
		Notes:   []string{"响应 key：{issues:[...]} / {issue:{...,comments:[...]}}。"},
		Related: []string{"pulls"},
	},
	"checks": {
		Topic:   "checks",
		Title:   "checks — 某 commit 的 CI",
		Summary: "查询某 commit 的 GitHub check-runs 与 combined status。",
		Usage: []string{
			"GET <BASE>/<owner>/<repo>/commits/<sha>/checks",
			"GET <BASE>/<owner>/<repo>/pulls/<n>/checks",
		},
		Examples: []string{
			"curl -sS 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/commits/<sha>/checks'",
			"curl -sS 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/pulls/7/checks'",
		},
		Notes: []string{
			"主要看 check_runs（conclusion=success/failure/skipped），combined_status 是旧 API 常为空。",
			"details_url 指向 GitHub Actions 具体 job。",
			"坏 sha → 400。",
		},
		Related: []string{"runs", "pulls"},
	},
	"runs": {
		Topic:    "runs",
		Title:    "runs — GitHub Actions 运行",
		Summary:  "查询最近 GitHub Actions runs，可按分支过滤。",
		Usage:    []string{"GET <BASE>/<owner>/<repo>/actions/runs?branch=<b>&limit=N"},
		Examples: []string{"curl -sS 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/actions/runs?limit=5&branch=main'"},
		Notes:    []string{"响应 {runs:[{databaseId,name,status,conclusion,headBranch,headSha,url}]}"},
		Related:  []string{"checks"},
	},
	"health": {
		Topic:    "health",
		Title:    "health — 健康检查",
		Summary:  "返回服务与 token / 白名单状态。",
		Usage:    []string{"GET <BASE>/health"},
		Examples: []string{"curl -sS 'https://wsl-8000.moonchan.xyz/health'"},
		Notes:    []string{"{ok, token, allow}"},
	},
	"force": {
		Topic:   "force",
		Title:   "force — 非快进保护",
		Summary: "禁止用 -f / --force-with-lease 把已有分支改写成非快进历史。",
		Usage:   []string{"git push -f origin <branch>   # 被拒"},
		Notes: []string{
			"实现：解析 receive-pack 命令，对已有分支更新用 git merge-base --is-ancestor 判断；非 ff → 403。",
			"新建分支 / fast-forward 放行。",
			"检查失败时跳过放行（best-effort），要 100% 强制可叠加 GitHub 分支保护。",
		},
		Related: []string{"delete", "push"},
	},
	"delete": {
		Topic:   "delete",
		Title:   "delete — 删除分支/tag 被禁止",
		Summary: "经代理的 git push 删除分支或 tag 一律 403。",
		Usage:   []string{"git push origin --delete <branch>   # 被拒", "git push origin :refs/tags/<tag>   # 被拒"},
		Notes: []string{
			"只会受影响的：经代理的 git 删除。管理员仍可直接对 GitHub 用 gh api -X DELETE。",
			"要彻底禁止，叠加 GitHub 规则集 / 分支保护。",
		},
		Related: []string{"force", "push"},
	},
	"auth": {
		Topic:   "auth",
		Title:   "auth — 鉴权模型",
		Summary: "代理用本机 gh token 向 github.com 注入鉴权；agent 侧全程匿名。",
		Usage:   []string{"# agent 无需任何凭据"},
		Notes: []string{
			"代理用 gh auth token（缓存 5 分钟）设 Authorization: Basic x-access-token:<token>；客户端 Authorization 被丢弃。",
			"git clone/push 路径不受 --api-token 限制。",
			"POST /commits 可选用 --api-token 保护。",
		},
		Related: []string{"whitelist"},
	},
	"whitelist": {
		Topic:   "whitelist",
		Title:   "whitelist — 仓库白名单",
		Summary: "只有白名单仓库可通过代理；其余 clone/push/查询一律 403。",
		Usage:   []string{"--allow <owner/repo>  （可重复；支持 owner/*）"},
		Notes: []string{
			"默认只放行 Hana-ame/agent。",
			"白名单是仓库粒度，不限制 patch 改哪些文件。",
		},
		Related: []string{"auth"},
	},
	"branches": {
		Topic:   "branches",
		Title:   "branches — 分支列表 / 详情 / 创建",
		Summary: "查询分支，或直接创建分支（从分支、commit sha 或 tag 起）。git push 新分支与 POST /commits 也会创建分支。",
		Usage: []string{
			"GET  <BASE>/<owner>/<repo>/branches?limit=N",
			"GET  <BASE>/<owner>/<repo>/branches/<name>",
			"POST <BASE>/<owner>/<repo>/branches?name=<new>&from=<branch|sha|tag>",
			`POST <BASE>/<owner>/<repo>/branches  JSON {"name":"new","from":"main"}`,
		},
		Examples: []string{
			"curl -sS 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/branches'",
			"curl -sS -X POST 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/branches?name=feat/new&from=main'",
			`curl -sS -X POST 'https://wsl-8000.moonchan.xyz/Hana-ame/agent/branches' -H 'Content-Type: application/json' -d '{"name":"feat/x","from":"<commit-sha>"}'`,
		},
		Notes: []string{
			"from 可省略，默认 --base（main）。",
			"from 支持分支名 / commit sha / tag。",
			"已存在的分支 -> 409；from 解析不到 -> 400；上游失败 -> 502。",
			"创建动作本身不做保护检查（保护只限制 push/commit 到该分支）。",
		},
		Related: []string{"commits", "push", "clone"},
	},
	"policy": {
		Topic:   "policy",
		Title:   "policy — 分支保护策略（gitignore 风格，按 owner/repo/branch）",
		Summary: "一行一条 owner/repo/branch-glob；! 前缀=反选(放行)；后匹配覆盖先匹配（gitignore 语义）。",
		Usage:   []string{"--protect-file ./branches.txt", "--protect 'Hana-ame/agent/main' --protect '!Hana-ame/agent/release/hotfix-*'", "--protect none   # 全部关闭"},
		Examples: []string{
			"# branches.txt\n*/*/main\nHana-ame/agent/release/*\nHana-ame/agent/release/**\n!Hana-ame/agent/release/hotfix-*\nacme/other/master",
		},
		Notes: []string{
			"每行 owner/repo/branch-glob：前面按 owner/repo 匹配（支持 * ?），后面是分支 glob。",
			"分支通配：* 单段（不跨 /），** 跨 /，? 单个非 / 字符。",
			"! 前缀 = 反选（放行）；顺序生效：后面的规则覆盖前面的（gitignore 风格）。",
			"没有规则匹配 -> 不保护；完全没配置时默认保护每个仓库的 --base 分支；--protect none 关闭。",
			"不同仓库互不影响；非法行启动时按 file:line 报错退出。",
			"被保护分支：git push / POST /commits 一律 403。",
		},
		Related: []string{"push", "commits", "delete"},
	},
	"errors": {
		Topic:   "errors",
		Title:   "errors — 常见错误码",
		Summary: "统一 JSON 错误。",
		Usage:   []string{"{ok:false,error,detail?,topics?}"},
		Notes: []string{
			"400 = 参数/patch/编号有问题；403 = 非白名单、force、删除；404 = 路径不存在/未知 help topic；502 = 上游 GitHub/gh 失败；401 = --api-token 未通过。",
		},
	},
}

type paramSpec struct {
	Flag    string   `json:"flag"`
	Type    string   `json:"type"`
	Default string   `json:"default"`
	Desc    string   `json:"desc"`
	Example []string `json:"example,omitempty"`
}

var paramSpecs = []paramSpec{
	{Flag: "--addr", Type: "string", Default: ":8000", Desc: "HTTP 监听地址。tunnel 会把 https://wsl-8000.moonchan.xyz 转发到这里。", Example: []string{":8000", "127.0.0.1:8000"}},
	{Flag: "--data", Type: "string", Default: "./data", Desc: "托管 clone 目录（POST /commits 与 force 检查用的本地仓库）。首次 SSH clone 约 1–2 分钟，之后复用。", Example: []string{"./data", "/var/lib/gitproxy/data"}},
	{Flag: "--tmp-dir", Type: "string", Default: "OS 临时目录", Desc: "临时 patch 文件目录。默认 os.TempDir()：Linux 为 /tmp，Windows 为 %TEMP%。补丁应用后立即删除。", Example: []string{"/tmp", "/var/tmp"}},
	{Flag: "--base", Type: "string", Default: "main", Desc: "POST /commits 新建分支的默认起点；也是 PR 的默认 base 分支。", Example: []string{"main", "master", "develop"}},
	{Flag: "--allow", Type: "string（可重复）", Default: "Hana-ame/agent", Desc: "仓库白名单，支持 owner/* 通配；值以 @ 开头时从文件读取。非白名单仓库 clone/push/POST/查询一律 403。", Example: []string{"Hana-ame/agent", "Hana-ame/*", "@allow.txt"}},
	{Flag: "--allow-file", Type: "string（可重复）", Default: "（空）", Desc: "从文本文件读取白名单，一行一个（空行与 # 注释忽略）。", Example: []string{"./allow.txt"}},
	{Flag: "--protect", Type: "string（可重复）", Default: "= --base（每个仓库的该分支）", Desc: "gitignore 风格一条规则：owner/repo/branch-glob；! 前缀=反选(放行)；后匹配覆盖先匹配；none 关闭。", Example: []string{"*/*/main", "Hana-ame/agent/release/**", "!Hana-ame/agent/release/hotfix-*", "none"}},
	{Flag: "--protect-file", Type: "string（可重复）", Default: "（空）", Desc: "gitignore 风格策略文件：一行一条 owner/repo/branch-glob，! 前缀=反选，后匹配覆盖先匹配。", Example: []string{"./branches.txt"}},
	{Flag: "--api-token", Type: "string", Default: "（空）", Desc: "设置后 POST /commits 需要 Authorization: Bearer <token>；git clone/push 路径不受此限制。", Example: []string{"s3cr3t"}},
}

var paramByName = func() map[string]paramSpec {
	m := make(map[string]paramSpec, len(paramSpecs))
	for _, p := range paramSpecs {
		m[strings.TrimLeft(p.Flag, "-")] = p
	}
	return m
}()

func paramNames() []string {
	out := make([]string, 0, len(paramSpecs))
	for _, p := range paramSpecs {
		out = append(out, strings.TrimLeft(p.Flag, "-"))
	}
	return out
}

func helpOverview(a *App) map[string]any {
	topics := make([]string, 0, len(helpTopics))
	for k := range helpTopics {
		topics = append(topics, k)
	}
	sort.Strings(topics)
	return map[string]any{
		"service": "gitproxy",
		"base":    "https://wsl-8000.moonchan.xyz",
		"summary": "把 GitHub 仓库经 HTTPS smart-HTTP 反代给多个 agent：支持标准 git clone/push/pull/merge、POST patch→commit→push→PR，以及 PR/CI/issue 查询。agent 侧零凭据，鉴权由代理侧注入 gh token。",
		"topics":  topics,
		"endpoints": []map[string]any{
			{"method": "POST", "path": "/<owner>/<repo>/commits", "desc": "提交 patch(diff) → commit → push，可选开 PR", "help": "/help/commits"},
			{"method": "GET", "path": "/<owner>/<repo>/branches", "desc": "分支列表", "help": "/help/branches"},
			{"method": "GET", "path": "/<owner>/<repo>/branches/<name>", "desc": "分支详情", "help": "/help/branches"},
			{"method": "POST", "path": "/<owner>/<repo>/branches", "desc": "创建分支（name, from=分支/commit/tag）", "help": "/help/branches"},
			{"method": "GET", "path": "/<owner>/<repo>/pulls", "desc": "PR 列表", "help": "/help/pulls"},
			{"method": "GET", "path": "/<owner>/<repo>/pulls/<n>", "desc": "PR 详情 + CI 汇总", "help": "/help/pulls"},
			{"method": "GET", "path": "/<owner>/<repo>/pulls/<n>/checks", "desc": "PR 头提交 check-runs", "help": "/help/checks"},
			{"method": "GET", "path": "/<owner>/<repo>/commits/<sha>/checks", "desc": "commit check-runs", "help": "/help/checks"},
			{"method": "GET", "path": "/<owner>/<repo>/actions/runs", "desc": "最近 Actions 运行", "help": "/help/runs"},
			{"method": "GET", "path": "/<owner>/<repo>/issues", "desc": "issue 列表", "help": "/help/issues"},
			{"method": "GET", "path": "/<owner>/<repo>/issues/<n>", "desc": "issue 详情 + comments", "help": "/help/issues"},
			{"method": "GET", "path": "/health", "desc": "健康检查", "help": "/help/health"},
			{"method": "GET", "path": "/help[/<topic>]", "desc": "本帮助", "help": "/help"},
			{"method": "GET/POST", "path": "/<owner>/<repo>[.git]/{info/refs,git-upload-pack,git-receive-pack}", "desc": "git smart-HTTP（clone/fetch/push）", "help": "/help/clone, /help/push"},
		},
		"params":       paramSpecs,
		"param_topics": paramNames(),
		"howto":        "GET /help/<topic> 查看端点/操作说明；GET /help/<param> 查看单个启动参数（如 /help/allow、/help/ --data 去掉横线）。",
	}
}

// --------------------------------------------------------------------------- //
//
//	branches API
//
// --------------------------------------------------------------------------- //
func parseBranchesPath(p string) (owner, repo string, ok bool) {
	rel := strings.TrimPrefix(p, "/")
	if !strings.HasSuffix(rel, "/branches") {
		return "", "", false
	}
	rel = strings.TrimSuffix(rel, "/branches")
	segs := strings.Split(rel, "/")
	if len(segs) != 2 || segs[0] == "" || segs[1] == "" {
		return "", "", false
	}
	return segs[0], strings.TrimSuffix(segs[1], ".git"), true
}

func (a *App) handleBranchList(w http.ResponseWriter, r *http.Request, owner, repo string) {
	perPage := clampInt(r.URL.Query().Get("limit"), 1, 100, 30)
	out, err := ghOut("api", fmt.Sprintf("repos/%s/%s/branches?per_page=%d", owner, repo, perPage))
	if err != nil {
		writeJSON(w, 502, errObj("gh api branches: "+err.Error()))
		return
	}
	writeRawJSON(w, `{"branches":`+strings.TrimSpace(out)+`}`)
}

func (a *App) handleBranchView(w http.ResponseWriter, r *http.Request, owner, repo, name string) {
	if name == "" {
		writeJSON(w, 400, errObj("missing branch name"))
		return
	}
	out, err := ghOut("api", fmt.Sprintf("repos/%s/%s/branches/%s", owner, repo, name))
	if err != nil {
		writeJSON(w, 502, errObj("gh api branch: "+err.Error()))
		return
	}
	writeRawJSON(w, `{"branch":`+strings.TrimSpace(out)+`}`)
}

// handleBranchCreate: 创建分支。body/query: name(必填), from(可选：分支名/commit sha/tag，默认 --base)
func (a *App) handleBranchCreate(w http.ResponseWriter, r *http.Request, owner, repo string) {
	name, from := "", ""
	if strings.Contains(r.Header.Get("Content-Type"), "application/json") {
		var j struct {
			Name string `json:"name"`
			From string `json:"from"`
		}
		_ = json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&j)
		name, from = j.Name, j.From
	} else {
		name = r.URL.Query().Get("name")
		from = r.URL.Query().Get("from")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		writeJSON(w, 400, errObj("missing 'name'"))
		return
	}
	if from == "" {
		from = a.base
	}
	// resolve from -> sha（分支 -> commit -> tag 都走 /commits/{ref} 兜底）
	sha := ""
	if out, err := ghOut("api", fmt.Sprintf("repos/%s/%s/git/ref/heads/%s", owner, repo, from)); err == nil {
		var ref struct {
			Object struct {
				Sha string `json:"sha"`
			} `json:"object"`
		}
		if json.Unmarshal([]byte(out), &ref) == nil {
			sha = ref.Object.Sha
		}
	}
	if sha == "" {
		if out, err := ghOut("api", fmt.Sprintf("repos/%s/%s/commits/%s", owner, repo, from)); err == nil {
			var c struct {
				Sha string `json:"sha"`
			}
			if json.Unmarshal([]byte(out), &c) == nil {
				sha = c.Sha
			}
		}
	}
	if sha == "" {
		writeJSON(w, 400, errObj("cannot resolve 'from' (branch/commit/tag): "+from))
		return
	}
	payload := fmt.Sprintf(`{"ref":"refs/heads/%s","sha":"%s"}`, name, sha)
	code, out := runCmdStdin("gh", []string{"api", "-X", "POST", fmt.Sprintf("repos/%s/%s/git/refs", owner, repo), "--input", "-"}, "", []byte(payload), 30*time.Second)
	if code != 0 {
		msg := strings.TrimSpace(out)
		status := 502
		if strings.Contains(msg, "already exists") || strings.Contains(msg, "Reference already exists") || strings.Contains(msg, "422") {
			status = 409
		}
		writeJSON(w, status, errObj("create branch: "+msg))
		return
	}
	writeRawJSON(w, `{"ok":true,"branch":{"name":`+jsonStr(name)+`,"sha":`+jsonStr(sha)+`},"created":`+strings.TrimSpace(out)+`}`)
}
