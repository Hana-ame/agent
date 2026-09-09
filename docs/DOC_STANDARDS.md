# Documentation Standards

Pinned formatting and authoring rules for this repository. Every rule below came
from a concrete problem found in this tree, and each one names the command or
check that enforces it. When you disagree with a rule, change this file in the
same commit that breaks it — do not let the practice drift silently.

## 1. Language

- **English** by default, including section titles.
- `docs/ARCHITECTURE.md` is the one exception: it is a Chinese narrative and
  stays Chinese.
- A section title may carry a Chinese annotation in parentheses for readers who
  search in Chinese, e.g. `## 2. All Execution Modes & Running Guide (运行方式全景指南)`.
  The annotation is the second part of the title, never a separate heading.
- Do not mix languages inside one paragraph. If a paragraph needs both, split it.

*Why:* the review record, the RFC and every code comment are English. Per-paragraph
mixing makes diffs unreadable.

## 2. Naming

- The **canonical** form is the short name that matches the JSON key it maps to:
  `--session` ↔ `"session"`, `--seed` ↔ `"seed"`, `--id` ↔ `"id"`,
  `--input`/`--output` ↔ `"input_vertex"`/`"output_vertex"`.
- A deprecated alias is allowed to stay, but it must be labelled as deprecated in
  the `--help` text *and* in the naming table. Four aliases are currently kept:
  `--edge-id`, `--seed-input`, `"session_id"`, `"seed_input"` (plus the lenient
  `"edge_type"` config key).
- **One source of truth:** the naming table in `README.md` (Quick Start §5).
  `agent.md` mirrors it. If a flag or key is renamed, both change in the same
  commit, and the alias test in `tests/test_edges_cli_type.py` stays green.

*Why:* four inconsistent names were fixed in one round; the churn was in the
tests and docs, not in the rename itself. Naming is cheap to change once and
expensive to keep ambiguous forever.

## 3. Counts and claims

- Never write `N tests passing` or `100% pass rate`.
- Write the command and the result of that command, e.g.
  `723 passed, 0 failed` next to `pytest tests/ -q -m "not live"`.
- A bare number with no command beside it is a claim that will go stale.
- The 6 `live` tests are always named as deselected; they need real API access.

*Why:* three stale `551 tests` claims survived several revisions;
`docs/archive/handoff-v4.md` still advertises 551 today.

## 4. Docs taxonomy

- Current documentation lives at `docs/*.md`. Historical documentation lives in
  `docs/archive/`.
- Archives are **records, not references**: never edit one in place to make it
  current. If it says something wrong now, note that in the index row.
- `docs/README.md` is the index. Every non-archived `docs/*.md` must appear in
  its Current table. The checker fails if one is missing.
- A doc that is not reachable from `README.md` or the index is a candidate for
  the archive — that is how three files were found.

## 5. Evidence hygiene

- No local-machine absolute paths: no `/home/<user>/`, no `/mnt/...`, no
  `file:///`. Link by repo-relative path, e.g.
  `[VertexStoreV4](../framework/vertex_v4.py)` from inside `docs/`.
- Do not quote another machine's layout as if it were this one.
- One file per document. Do not keep byte-identical copies in two places.

*Why:* `agent.md` linked to `file:///home/gekkasayu/vertex_edge_agent/...`, and
`docs/review_report.md` was byte-identical to a copy in `examples/`.

## 6. Every command in a document must be executed before it is committed

Run it, paste the real output, then commit. If you cannot run it, do not put it
in a fenced block that looks runnable.

*Why:* this session caught three such errors before commit — `script` wrongly
nested under `settings` (the code edge silently passes input through), `/api/edge-types`
described as listing dynamic script specs (it returns registered types only), and
`run_from_manifest(print_events=True)` with a parameter that does not exist.

## 7. Every API claim must be checked against the code

Use `inspect.signature`, the dataclass fields, the enum members, and the route
handler. Do not paraphrase from memory.

*Why:* `load_v4_manifest` is not exported from `framework.graphs.loader`; it
lives in `framework.run_v4`. The first draft of the Quick Start imported it from
the wrong module.

## 8. Markdown mechanics

- One `# H1` per file. `README.md` uses numbered `## 1.` … `## 8.` sections;
  `###` for subsections. Do not insert an unnumbered section between two
  numbered ones unless it is labelled as a quick-start block.
- Fences carry a language tag when one applies: `python`, `bash`, `json`.
  An untagged fence is for **command output only** — that is how a reader tells
  input from output.
- Tables: an alignment row of `:---`, no trailing separator lines, and no stray
  separator fragment left in the prose below the table.
- Code samples must be self-contained: if a snippet uses `graph` or `store`, say
  where they come from above it.
- Relative links only, and they must resolve — including from the file's own
  directory, so `docs/README.md` links to `ARCHITECTURE.md`, not
  `docs/ARCHITECTURE.md`. When illustrating markdown syntax itself, keep the
  sample inside inline code (`` `# [Title](Link)` ``) so it is not read as a
  real link.

## 9. Commit messages

- `<type>: <imperative summary>` — `docs:`, `test:`, `feat:`, `fix:`, `refactor:`,
  `ci:`.
- The body explains *why*, and names what was verified and by what command.
- Docs-only commits end with `Docs only.` so the reviewer knows no test run is
  implied.
- Archive and index changes are described file by file; a rename is not the same
  thing as a deletion.

## 10. Enforcement

```bash
python3 scripts/check_docs.py        # exit 1 on failure, 0 when clean
```

Runs in CI as part of the default job. It checks: fence balance, relative link
resolution (archives exempt), local absolute paths, stale count phrasing, index
coverage, and untagged fences (warning only).

**Not checkable mechanically, so a human must still look:**

1. Whether the commands were actually run (§6).
2. Whether API claims match the code (§7).
3. Whether the prose is clear and the right document is the one being read.
4. Whether something belongs in the archive rather than the current set (§4).
