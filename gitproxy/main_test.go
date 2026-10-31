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
