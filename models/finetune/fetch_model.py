"""Pre-download the base model so the first training run does not stall on I/O.

    python -m finetune.fetch_model
"""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

from finetune.common import MODEL_ID


def main() -> int:
    print(f"downloading {MODEL_ID} ...")
    path = snapshot_download(
        MODEL_ID,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "merges.txt", "vocab.json"],
        max_workers=4,
    )
    print("cached at:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
