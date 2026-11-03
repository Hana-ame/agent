package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --------------------------------------------------------------------------- //
//
//	glob / branch policy
//
// --------------------------------------------------------------------------- //
func TestGlobToRegexp(t *testing.T) {
	cases := []struct {
		pat, name string
		want      bool
	}{
		{"main", "main", true},
		{"main", "main/x", false},
		{"release/*", "release/v1", true},
		{"release/*", "release/v1/x", false},
		{"release/**", "release/v1", true},
		{"release/**", "release/v1/x", true},
		{"release/**", "main", false},
		{"feat-?", "feat-a", true},
		{"feat-?", "feat-ab", false},
		{"feat-?", "feat-/", false},
		{"main.v2", "main.v2", true},
		{"main.v2", "mainXv2", false},
	}
	for _, c := range cases {
		re, err := globToRegexp(c.pat)
		if err != nil {
			t.Fatalf("globToRegexp(%q) error: %v", c.pat, err)
		}
		if got := re.MatchString(c.name); got != c.want {
			t.Errorf("pattern %q vs %q = %v, want %v", c.pat, c.name, got, c.want)
		}
	}
}

func TestBranchPolicyIsProtected(t *testing.T) {
	bp, err := loadBranchPolicy(nil, []string{"main", "release/*", "release/**"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if bp.isProtected("feat/x") {
		t.Error("feat/x should not be protected")
	}
	// allow overrides protect
	bp2 := &branchPolicy{}
	for _, p := range []string{"release/*", "release/**"} {
		re, _ := globToRegexp(p)
		bp2.protect = append(bp2.protect, re)
	}
	re, _ := globToRegexp("release/hotfix-*")
	bp2.allow = append(bp2.allow, re)
	if !bp2.isProtected("release/v1") {
		t.Error("release/v1 should be protected")
	}
	if bp2.isProtected("release/hotfix-1") {
		t.Error("release/hotfix-1 should be allowed (allow overrides protect)")
	}
	var nilPol *branchPolicy
	if nilPol.isProtected("main") {
		t.Error("nil policy should protect nothing")
	}
}

func TestParseBranchPolicyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "branches.txt")
	content := `# comment line
[protect]
main
release/*
release/**

[allow]
release/hotfix-*
inline-allow-x
`
	// NOTE: 'inline-allow-x' sits in the [allow] section -> allow entry
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	protect, allow, err := parseBranchPolicyFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(protect, ",") != "main,release/*,release/**" {
		t.Errorf("protect = %v", protect)
	}
	if strings.Join(allow, ",") != "release/hotfix-*,inline-allow-x" {
		t.Errorf("allow = %v", allow)
	}

	// invalid section
	bad := filepath.Join(dir, "bad.txt")
	_ = os.WriteFile(bad, []byte("[nope]\nmain\n"), 0o644)
	if _, _, err := parseBranchPolicyFile(bad); err == nil {
		t.Error("expected error for unknown section")
	}

	// missing file
	if _, _, err := parseBranchPolicyFile(filepath.Join(dir, "nope.txt")); err == nil {
		t.Error("expected error for missing file")
	}
}

func TestLoadBranchPolicyDefaults(t *testing.T) {
	// no files, no inline -> protect the base branch
	bp, err := loadBranchPolicy(nil, nil, "main")
	if err != nil {
		t.Fatal(err)
	}
	if !bp.isProtected("main") {
		t.Error("default should protect base branch")
	}
	if bp.isProtected("feat/x") {
		t.Error("default should not protect feat/x")
	}

	// "none" disables all protection
	bp, err = loadBranchPolicy(nil, []string{"none"}, "main")
	if err != nil {
		t.Fatal(err)
	}
	if bp.isProtected("main") {
		t.Error("none should disable protection")
	}

	// inline globs
	bp, err = loadBranchPolicy(nil, []string{"release/*"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if !bp.isProtected("release/v1") || bp.isProtected("release/v1/x") {
		t.Error("inline release/* should protect one segment only")
	}
}

// --------------------------------------------------------------------------- //
//
//	pkt-line parsing
//
// --------------------------------------------------------------------------- //
func TestParseHexLen(t *testing.T) {
	cases := []struct {
		in   string
		want int
		ok   bool
	}{
		{"0000", 0, true}, // flush
		{"0035", 53, true},
		{"00c7", 199, true},
		{"0003", 0, false}, // too small (but non-zero)
		{"zzzz", 0, false},
		{"000", 0, false}, // wrong length
	}
	for _, c := range cases {
		got, err := parseHexLen([]byte(c.in))
		if c.ok && err != nil {
			t.Errorf("parseHexLen(%q) unexpected error: %v", c.in, err)
		}
		if !c.ok && err == nil {
			t.Errorf("parseHexLen(%q) expected error", c.in)
		}
		if c.ok && got != c.want {
			t.Errorf("parseHexLen(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestParseReceiveCmds(t *testing.T) {
	oldSHA := strings.Repeat("a", 40)
	newSHA := strings.Repeat("b", 40)
	var body []byte
	body = append(body, pktLine([]byte("shallow "+oldSHA+"\n"))...)
	body = append(body, pktLine([]byte(oldSHA+" "+newSHA+" refs/heads/main\x00 report-status-v2"))...)
	body = append(body, []byte("0000")...)
	body = append(body, []byte("PACKpayload")...)

	cmds, flushOff, ok := parseReceiveCmds(body)
	if !ok {
		t.Fatal("expected ok")
	}
	if len(cmds) != 1 {
		t.Fatalf("cmds = %d, want 1", len(cmds))
	}
	c := cmds[0]
	if c.old != oldSHA || c.new != newSHA || c.ref != "refs/heads/main" {
		t.Errorf("cmd = %+v", c)
	}
	if flushOff <= 0 {
		t.Errorf("flushOff = %d", flushOff)
	}

	// truncate the body -> must not claim ok (command list incomplete)
	if _, _, ok := parseReceiveCmds(body[:10]); ok {
		t.Error("expected !ok for truncated body")
	}
}

// --------------------------------------------------------------------------- //
//
//	routing helpers
//
// --------------------------------------------------------------------------- //
func TestParseGitPath(t *testing.T) {
	cases := []struct {
		path, owner, repo, service string
		ok                         bool
	}{
		{"/Hana-ame/agent.git/info/refs", "Hana-ame", "agent", "info/refs", true},
		{"/Hana-ame/agent/git-upload-pack", "Hana-ame", "agent", "git-upload-pack", true},
		{"/o/r/git-receive-pack", "o", "r", "git-receive-pack", true},
		{"/o/r/nope", "", "", "", false},
		{"/o/r", "", "", "", false},
		{"/o/r/x/y", "", "", "", false},
	}
	for _, c := range cases {
		o, r, s, ok := parseGitPath(c.path)
		if ok != c.ok || o != c.owner || r != c.repo || s != c.service {
			t.Errorf("parseGitPath(%q) = (%q,%q,%q,%v)", c.path, o, r, s, ok)
		}
	}
}

func TestParseCommitPath(t *testing.T) {
	cases := []struct {
		path, owner, repo string
		ok                bool
	}{
		{"/commits", "", "", true},
		{"/Hana-ame/agent/commits", "Hana-ame", "agent", true},
		{"/Hana-ame/agent.git/commits", "Hana-ame", "agent", true},
		{"/commit", "", "", false}, // alias removed
		{"/Hana-ame/agent/commit", "", "", false},
		{"/a/b/commits/x", "", "", false},
	}
	for _, c := range cases {
		o, r, ok := parseCommitPath(c.path)
		if ok != c.ok || o != c.owner || r != c.repo {
			t.Errorf("parseCommitPath(%q) = (%q,%q,%v)", c.path, o, r, ok)
		}
	}
}

func TestSplitMetaPath(t *testing.T) {
	cases := []struct {
		path, owner, repo, rest string
		ok                      bool
	}{
		{"/Hana-ame/agent/pulls", "Hana-ame", "agent", "pulls", true},
		{"/Hana-ame/agent.git/pulls/7/checks", "Hana-ame", "agent", "pulls/7/checks", true},
		{"/Hana-ame/agent", "Hana-ame", "agent", "", true},
		{"/health", "", "", "", false},
		{"/", "", "", "", false},
	}
	for _, c := range cases {
		o, r, rest, ok := splitMetaPath(c.path)
		if ok != c.ok || o != c.owner || r != c.repo || rest != c.rest {
			t.Errorf("splitMetaPath(%q) = (%q,%q,%q,%v)", c.path, o, r, rest, ok)
		}
	}
}

func TestMatchRepo(t *testing.T) {
	cases := []struct {
		pat, full string
		want      bool
	}{
		{"Hana-ame/agent", "Hana-ame/agent", true},
		{"Hana-ame/agent", "Hana-ame/other", false},
		{"Hana-ame/*", "Hana-ame/agent", true},
		{"Hana-ame/*", "Hana-ame/whatever", true},
		{"Hana-ame/*", "torvalds/linux", false},
	}
	for _, c := range cases {
		if got := matchRepo(c.pat, c.full); got != c.want {
			t.Errorf("matchRepo(%q,%q) = %v, want %v", c.pat, c.full, got, c.want)
		}
	}
}

func TestClampInt(t *testing.T) {
	cases := []struct {
		in       string
		lo, hi   int
		def, exp int
	}{
		{"", 1, 100, 20, 20},
		{"-5", 1, 100, 20, 1},
		{"999", 1, 100, 20, 100},
		{"7", 1, 100, 20, 7},
		{"abc", 1, 100, 20, 20},
	}
	for _, c := range cases {
		if got := clampInt(c.in, c.lo, c.hi, c.def); got != c.exp {
			t.Errorf("clampInt(%q) = %d, want %d", c.in, got, c.exp)
		}
	}
}

func TestDedupeStrings(t *testing.T) {
	got := dedupeStrings([]string{"a", "b", "a", "c", "b"})
	if strings.Join(got, ",") != "a,b,c" {
		t.Errorf("dedupeStrings = %v", got)
	}
}

func TestReadAllowFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "allow.txt")
	content := "# comment\nHana-ame/agent\n\n torvalds/* # trailing\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	lines, err := readAllowFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(lines, ",") != "Hana-ame/agent,torvalds/*" {
		t.Errorf("readAllowFile = %v", lines)
	}
	if _, err := readAllowFile(filepath.Join(dir, "missing.txt")); err == nil {
		t.Error("expected error for missing allow file")
	}
}

// --------------------------------------------------------------------------- //
//
//	GET /allow + clone hint
//
// --------------------------------------------------------------------------- //
func TestBranchPolicyPatterns(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "branches.txt")
	content := "# comment\n[protect]\nmain\nrelease/*\n\n[allow]\nrelease/hotfix-*\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	pol, err := loadBranchPolicy([]string{path}, []string{"main"}, "main")
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(pol.protectPatterns, ","); got != "main,release/*" {
		t.Errorf("protectPatterns = %q", got)
	}
	if got := strings.Join(pol.allowPatterns, ","); got != "release/hotfix-*" {
		t.Errorf("allowPatterns = %q", got)
	}

	def, err := loadBranchPolicy(nil, nil, "main")
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(def.protectPatterns, ","); got != "main" {
		t.Errorf("default protectPatterns = %q", got)
	}

	none, err := loadBranchPolicy(nil, []string{"none"}, "main")
	if err != nil {
		t.Fatal(err)
	}
	if len(none.protectPatterns) != 0 || len(none.allowPatterns) != 0 {
		t.Errorf("none policy should have no patterns")
	}
}

type allowResponse struct {
	OK           bool     `json:"ok"`
	Repos        []string `json:"repos"`
	Base         string   `json:"base"`
	BranchPolicy struct {
		Protect []string `json:"protect"`
		Allow   []string `json:"allow"`
	} `json:"branch_policy"`
}

func TestHandleAllow(t *testing.T) {
	pol, err := loadBranchPolicy(nil, []string{"main", "release/*"}, "main")
	if err != nil {
		t.Fatal(err)
	}
	a := &App{base: "main", allow: []string{"Hana-ame/agent", "owner/*"}, policy: pol}

	rec := httptest.NewRecorder()
	a.router(rec, httptest.NewRequest(http.MethodGet, "/allow", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", rec.Code, rec.Body.String())
	}
	var got allowResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.OK || strings.Join(got.Repos, ",") != "Hana-ame/agent,owner/*" {
		t.Errorf("got %+v", got)
	}
	if strings.Join(got.BranchPolicy.Protect, ",") != "main,release/*" {
		t.Errorf("protect = %v", got.BranchPolicy.Protect)
	}
	if got.Base != "main" {
		t.Errorf("base = %q", got.Base)
	}
}

func TestCloneDepthHint(t *testing.T) {
	const hint = "无特殊说明，使用 --depth 1 --single-branch"
	clone := helpTopics["clone"]
	if !strings.Contains(strings.Join(clone.Notes, "\n"), hint) {
		t.Errorf("clone help missing hint")
	}
	if !strings.Contains(strings.Join(clone.Usage, "\n"), "--single-branch") {
		t.Errorf("clone usage missing --single-branch")
	}
	if got := infoObj(&App{})["usage_hint"]; got != hint {
		t.Errorf("infoObj usage_hint = %v", got)
	}
	if got := helpOverview(&App{})["hint"]; got != hint {
		t.Errorf("helpOverview hint = %v", got)
	}
}

func TestPushHint(t *testing.T) {
	const hint = "所有改动都要 push"
	if got := infoObj(&App{})["hint"]; got != hint {
		t.Errorf("infoObj hint = %v", got)
	}
	if got := helpOverview(&App{})["push_hint"]; got != hint {
		t.Errorf("helpOverview push_hint = %v", got)
	}
	if !strings.Contains(strings.Join(helpTopics["push"].Notes, "\n"), hint) {
		t.Errorf("push help missing hint")
	}
	if !strings.Contains(strings.Join(helpTopics["commits"].Notes, "\n"), hint) {
		t.Errorf("commits help missing hint")
	}
}

func TestCIHint(t *testing.T) {
	want := "不要本地编译测试；全部 push 后查看 CI（commits/<sha>/checks · pulls/<n>/checks · actions/runs?branch=<b>&limit=N）"
	if got := infoObj(&App{})["ci_hint"]; got != want {
		t.Errorf("infoObj ci_hint = %v", got)
	}
	if !strings.Contains(helpOverview(&App{})["ci_hint"].(string), "不要本地编译测试") {
		t.Errorf("helpOverview ci_hint missing")
	}
	if !strings.Contains(strings.Join(helpTopics["checks"].Notes, "\n"), "不要本地编译测试") {
		t.Errorf("checks help missing")
	}
	if !strings.Contains(strings.Join(helpTopics["runs"].Notes, "\n"), "不要本地编译测试") {
		t.Errorf("runs help missing")
	}
}
