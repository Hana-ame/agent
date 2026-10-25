#!/usr/bin/env python3
import json, os
data = json.loads(os.environ["DAG8_INPUTS"])
text = data.get("task", "")
print(json.dumps({"chars": len(text), "words": len(text.split()),
                  "lines": text.count("\n") + 1}, ensure_ascii=False))
