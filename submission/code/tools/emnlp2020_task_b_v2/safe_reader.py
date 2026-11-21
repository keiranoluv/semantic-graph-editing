"""Same v1 non-executing grammar/seccomp jail, compact range CLI for full splits."""

import json
import sys
from pathlib import Path

# -I ignores PYTHONPATH: load only repository-owned reader before confinement.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.emnlp2020_graph_modification_v1.safe_pickle import confine, scan  # noqa: E402

if __name__ == "__main__":
    confine(sys.argv[1])
    content = sys.stdin.buffer.read(32 * 1024**2 + 1)
    if len(content) > 32 * 1024**2:
        raise ValueError("member_byte_budget")
    result = scan(content, range(int(sys.argv[2])))
    sys.stdout.write(json.dumps(result, ensure_ascii=True))
