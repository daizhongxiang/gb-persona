from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import compute_all_results, verify_against_paper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the 128- and 64-persona results reported in the paper."
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing config.json, data/, and results/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduced_results"),
        help="Directory for newly computed tables and selections.",
    )
    args = parser.parse_args()

    root = args.repository_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results, selections = compute_all_results(root)
    verify_against_paper(results, root / "results" / "paper_results.csv")
    results.to_csv(output_dir / "reproduced_results.csv", index=False)
    (output_dir / "reproduced_selections.json").write_text(
        json.dumps(selections, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Verified all 64 reported curve points and AUECs; "
        f"outputs written to {output_dir}"
    )


if __name__ == "__main__":
    main()

