import json
import subprocess
import sys
from pathlib import Path


def main():
    # Prepare demo table metadata
    table_meta = {
        "table_name": "sensor_data",
        "columns": [
            {"name": "device_id", "type": "STRING"},
            {"name": "temperature", "type": "DOUBLE"},
            {"name": "ts", "type": "TIMESTAMP"},
        ],
        "partition_by": "DAY",
        "wal_enabled": True,
    }

    meta_path = Path("data/demo_table.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(table_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    question = "How can I efficiently query the average temperature per device over the last hour?"

    # Run RAG
    cmd = [
        sys.executable,
        "rag.py",
        "--table",
        str(meta_path),
        "--question",
        question,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


