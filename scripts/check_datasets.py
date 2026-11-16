"""Audit the three releases before running the benchmark.

Prints, per database, the number of rows in the released CSV, the number of
items the loader returns, the generator counts and any score row that has no
matching image file.  The output is the record behind the image counts quoted
in the paper.

    python scripts/check_datasets.py > results/bench/dataset_audit.txt
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import datasets  # noqa: E402


def rows_in_csv(name):
    base = datasets.root(name)
    csv_name = {"agiqa3k": "data.csv", "aigciqa2023": "data.csv",
                "agiqa1k": "AIGC_all.csv"}[name]
    return len(datasets._rows(os.path.join(base, csv_name)))


def main():
    print(f"{'database':<12}{'csv rows':>9}{'loaded':>8}{'missing':>9}")
    for name in ("agiqa3k", "aigciqa2023", "agiqa1k"):
        items = datasets.load(name)
        missing = datasets.SKIPPED.get(name, [])
        print(f"{name:<12}{rows_in_csv(name):>9}{len(items):>8}{len(missing):>9}")
        counts = collections.Counter(i["generator"] for i in items)
        print("  generators: " + ", ".join(
            f"{g} {n}" for g, n in sorted(counts.items())))
        if missing:
            print("  rows without an image file: " + ", ".join(missing))
        keys = [i["key"] for i in items]
        dupes = [k for k, n in collections.Counter(keys).items() if n > 1]
        if dupes:
            print("  duplicate keys: " + ", ".join(dupes))
        mos = [i["mos"] for i in items]
        print(f"  MOS range: {min(mos):.3f} to {max(mos):.3f}")


if __name__ == "__main__":
    main()
