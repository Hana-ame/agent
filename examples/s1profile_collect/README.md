# Forum Profile Collect (Auxiliary Data Directory)

> Auxiliary directory (gitignored data). Documented following the "Problem / Solution / Changes / Verification" format.

## Problem
Forum crawling and user analysis required an auxiliary data collection pipeline producing large volume output that should not be tracked in version control.

## Solution
Maintain a separate collection directory where all JSON files under `data/` are ignored via `.gitignore` (`examples/s1profile_collect/data/`), avoiding repository bloat.

## Changes
- `.gitignore`: Added `examples/s1profile_collect/data/` (all JSON data files within this path are ignored).

## Verification
- **Test Plan**: Verify data directory is ignored by git.
- **Method**: `git status --ignored --short | grep s1profile_collect`.
- **Result**: `data/` appears in the ignored files list; repository history remains free of raw crawled artifacts.
