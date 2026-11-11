"""
    python -m order_effect_eval.cli validate                  # synthetic methodology check
    python -m order_effect_eval.cli report data/pilot_results.csv
"""
from __future__ import annotations

import argparse
import sys

from order_effect_eval.agreement import bootstrap_ac1_ci
from order_effect_eval.flip import flip_rate
from order_effect_eval.simulate import generate
from order_effect_eval.report import load_pilot_csv, load_special_cases, render_markdown, write_report


def run_validate(n_items: int, seed: int) -> None:
    print(f"Synthetic validation (n={n_items} items per scenario)\n")
    for effect in (0.0, 0.3, 0.6):
        data = generate(n_items=n_items, order_effect_size=effect, seed=seed)
        truth = data["true_label"]
        rf, jf = data["reasoning_first"], data["judgment_first"]

        ac1_rf, *_ = bootstrap_ac1_ci(truth, rf, n_boot=1000, seed=seed)
        ac1_jf, *_ = bootstrap_ac1_ci(truth, jf, n_boot=1000, seed=seed)
        rate, _ = flip_rate(rf, jf)

        print(f"injected order_effect_size = {effect}")
        print(f"  AC1 reasoning-first vs truth : {ac1_rf:.3f}")
        print(f"  AC1 judgment-first  vs truth : {ac1_jf:.3f}")
        print(f"  flip rate                    : {rate:.3f}")
        print()
    print("Expected pattern: as order_effect_size rises, flip rate rises and "
          "AC1(reasoning-first) drops below AC1(judgment-first). If that pattern "
          "doesn't hold, the pipeline has a bug -- this is the check that it doesn't.")


def run_report(csv_path: str, out_path: str, special_cases_path: str | None = None) -> None:
    rows = load_pilot_csv(csv_path)
    if not rows:
        print(f"No rows found in {csv_path}")
        sys.exit(1)
    special_cases = load_special_cases(special_cases_path) if special_cases_path else []
    markdown = render_markdown(rows, special_cases)
    write_report(out_path, markdown)
    print(markdown)
    print(f"\nreport written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="run the synthetic methodology check")
    p_validate.add_argument("--n-items", type=int, default=200)
    p_validate.add_argument("--seed", type=int, default=0)

    p_report = sub.add_parser("report", help="render a markdown report from a real pilot CSV")
    p_report.add_argument("csv_path")
    p_report.add_argument("--out", default="reports/pilot_report.md")
    p_report.add_argument("--special-cases", default="data/special_cases.csv")

    args = parser.parse_args()
    if args.command == "validate":
        run_validate(args.n_items, args.seed)
    elif args.command == "report":
        run_report(args.csv_path, args.out, args.special_cases)


if __name__ == "__main__":
    main()
