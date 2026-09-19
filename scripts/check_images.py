"""
Find and delete unreadable (truncated) images left behind by interrupted runs.

  python scripts/check_images.py           # report and delete
  python scripts/check_images.py --dry-run # report only

Scans: synthetic, synthetic_refined, feature_cache (dihedral variants, crops). Deleted files are
regenerated automatically by the next run of the script that made them.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = common.load_config()
    roots = [cfg["paths"]["synthetic_dir"], cfg["paths"]["synthetic_refined_dir"], cfg["paths"]["cache_dir"]]
    checked = bad = 0
    for root in roots:
        for dp, _, files in os.walk(root):
            for f in files:
                p = os.path.join(dp, f)
                if f.endswith(".tmp.png"):
                    print(f"  leftover temp file: {p}")
                    if not args.dry_run:
                        os.remove(p)
                    continue
                if not f.lower().endswith(common.IMG_EXT):
                    continue
                checked += 1
                if not common.is_valid_image(p):
                    bad += 1
                    print(f"  {'would delete' if args.dry_run else 'deleted'}: {p}")
                    if not args.dry_run:
                        os.remove(p)
    print(f"\nchecked {checked} images, {bad} unreadable" + (" (not deleted, dry run)" if args.dry_run else " (deleted)"))
    if bad and not args.dry_run:
        print("Re-run script 3 for the affected seed(s) to regenerate refined images; caches regenerate on their own.")


if __name__ == "__main__":
    main()
