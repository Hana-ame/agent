#!/usr/bin/env bash
# Finalize step: after the continuous curve finishes (all 205-220 results exist),
#  1. patch builder rows with real metrics (backfill_results.py)
#  2. run the Excel builder (regenerates all Excels AND all JSON configs)
#  3. flip status=completed on 205-220 configs
#
# IMPORTANT (2026-09): We intentionally do NOT restore configs from config_backup/.
# The Excel builder (expand_and_update_all_excels.py) now generates ALL 474 configs
# (including the 221-440 32-vocab counterparts) directly from the experiment rows.
# The old `rm -rf configs && cp config_backup/*.json` step would have DELETED the
# 219 vocab32 configs (config_backup only held 254 stale files). Configs are now
# self-managed by the builder, so no external restore is needed.
set -e
cd "$(dirname "$0")"

echo "=== [1/3] Patch builder rows with real metrics ==="
python3 backfill_results.py

echo "=== [2/3] Run Excel builder (also regenerates all JSON configs) ==="
# openpyxl is required here; match the README SOP: `uv run --with openpyxl python3 ...`
# (falls back to plain python3 if uv is unavailable but openpyxl is on the path).
if command -v uv >/dev/null 2>&1; then
  PYTHONPATH="$PWD" uv run --with openpyxl python3 expand_and_update_all_excels.py
else
  PYTHONPATH="$PWD" python3 expand_and_update_all_excels.py
fi

echo "=== [3/3] Flip 205-220 configs to completed ==="
python3 - <<'EOF'
import json, glob, os
cfg_dir = 'additive-rand-transformer/configs'
for seq in range(205, 221):
    matches = glob.glob(os.path.join(cfg_dir, f'{seq}_*.json'))
    if not matches:
        print(f'  [warn] no config for {seq}')
        continue
    for p in matches:
        c = json.load(open(p))
        c['status'] = 'completed'
        json.dump(c, open(p, 'w'), ensure_ascii=False, indent=2)
print('step configs 205-220 marked completed')
EOF

echo "=== done ==="
