from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "test.txt"
    p.write_text("Hello pathlib")
    print("Read back:", p.read_text())
    print("Exists:", p.exists())
    print("Size:", p.stat().st_size)
