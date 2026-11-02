package main

import (
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

func writePolicy(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "branches.txt")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func writePolicyFile(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "branches.txt")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestBranchPolicyGitignore(t *testing.T) {
	path := writePolicyFile(t, "# gitignore-style\n*/*/main\nHana-ame/agent/release/*\nHana-ame/agent/release/**\n!Hana-ame/agent/release/hotfix-*\nacme/other/master\n")
	bp, err := loadBranchPolicy([]string{path}, nil, "main")
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		owner, repo, branch string
		want                bool
	}{
		{"Hana-ame", "agent", "main", true}, // */*/main
		{"any", "repo", "main", true},       // */*/main
		{"Hana-ame", "agent", "release/v1", true},
		{"Hana-ame", "agent", "release/v1/x", true},      // ** 跨 /
		{"Hana-ame", "agent", "release/hotfix-1", false}, // ! 反选
		{"Hana-ame", "agent", "feat/x", false},
		{"acme", "other", "master", true},
		{"acme", "other", "main", true}, // */*/main 也命中
		{"acme", "other", "dev", false},
	}
	for _, c := range cases {
		if got := bp.isProtected(c.owner, c.repo, c.branch); got != c.want {
			t.Errorf("isProtected(%s/%s,%s) = %v, want %v", c.owner, c.repo, c.branch, got, c.want)
		}
	}
}

// gitignore 关键语义：后匹配的规则覆盖前面的。
func TestBranchPolicyLastMatchWins(t *testing.T) {
	// 反选在后 -> 放行
	bp, err := loadBranchPolicy(nil, []string{"Hana-ame/agent/**", "!Hana-ame/agent/release/**"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if bp.isProtected("Hana-ame", "agent", "release/v1") {
		t.Error("later !rule should unprotect release/v1")
	}
	if !bp.isProtected("Hana-ame", "agent", "main") {
		t.Error("main should still be protected by **")
	}
	// 反选在前，保护在后 -> 保护
	bp, _ = loadBranchPolicy(nil, []string{"!Hana-ame/agent/release/**", "Hana-ame/agent/**"}, "")
	if !bp.isProtected("Hana-ame", "agent", "release/v1") {
		t.Error("later protect should re-protect release/v1")
	}
}

func TestBranchPolicyDefaultAndNone(t *testing.T) {
	bp, err := loadBranchPolicy(nil, nil, "main")
	if err != nil {
		t.Fatal(err)
	}
	if !bp.isProtected("any", "repo", "main") || bp.isProtected("any", "repo", "feat") {
		t.Error("default should protect main on every repo only")
	}
	bp, _ = loadBranchPolicy(nil, []string{"none"}, "main")
	if bp.isProtected("any", "repo", "main") {
		t.Error("none should disable all protection")
	}
	// 有配置时没有隐式默认
	bp, _ = loadBranchPolicy(nil, []string{"acme/other/master"}, "main")
	if bp.isProtected("any", "repo", "main") {
		t.Error("no implicit default once rules are configured")
	}
}

func TestBranchPolicyInline(t *testing.T) {
	bp, err := loadBranchPolicy(nil, []string{"Hana-ame/agent/main", "acme/other/release/**"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if !bp.isProtected("Hana-ame", "agent", "main") || !bp.isProtected("acme", "other", "release/v1") {
		t.Error("inline rules not applied")
	}
	if bp.isProtected("Hana-ame", "agent", "release/v1") {
		t.Error("unrelated repo rule must not apply")
	}
}

func TestBranchPolicyWildcards(t *testing.T) {
	bp, err := loadBranchPolicy(nil, []string{"*/agent/main", "Hana-*/other/**"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if !bp.isProtected("any", "agent", "main") {
		t.Error("owner * wildcard")
	}
	if !bp.isProtected("Hana-ame", "other", "release/v1") {
		t.Error("owner glob Hana-* should match")
	}
	if bp.isProtected("other", "other", "release/v1") {
		t.Error("Hana-* must not match other")
	}
}

func TestBranchPolicyInvalidRules(t *testing.T) {
	for _, bad := range []string{"main", "Hana-ame/agent", "Hana-ame/agent/", "/agent/main", "Hana-ame//main"} {
		if _, err := parseBranchRule(bad, "test:1"); err == nil {
			t.Errorf("expected error for %q", bad)
		}
	}
	path := writePolicyFile(t, "main\n")
	if _, err := loadBranchPolicy([]string{path}, nil, "main"); err == nil {
		t.Error("expected error for bad policy file line")
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
