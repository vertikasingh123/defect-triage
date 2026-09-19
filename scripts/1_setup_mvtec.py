"""
Script 1: Locate / extract / verify the MVTec AD category.

  python scripts/1_setup_mvtec.py

MVTec AD is free but requires registration:
  https://www.mvtec.com/company/research/datasets/mvtec-ad
Download mvtec_ad.tar.xz into data/mvtec_ad/ (or the per-category archive, e.g. screw.tar.xz),
then run this script. If the category folder is already extracted, it just verifies it.
"""

import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def try_extract(mvtec_dir, category):
    """Extract any tar archive found in mvtec_dir that could contain the category."""
    archives = sorted(f for f in os.listdir(mvtec_dir) if f.endswith((".tar.xz", ".tar", ".tar.gz", ".tgz")))
    if not archives:
        return False
    for name in archives:
        path = os.path.join(mvtec_dir, name)
        print(f"extracting {name} (this can take a few minutes)...")
        try:
            with tarfile.open(path, "r:*") as tar:
                tar.extractall(mvtec_dir)
        except Exception as e:
            print(f"  ! could not extract {name}: {e}")
            continue
        if os.path.isdir(os.path.join(mvtec_dir, category)):
            return True
    return os.path.isdir(os.path.join(mvtec_dir, category))


def verify(cfg):
    p = common.mvtec_paths(cfg)
    ok = True
    train_good = common.list_images(p["train_good"])
    test_good = common.list_images(p["test_good"])
    classes = common.defect_classes(cfg)

    print(f"\ncategory      : {p['category']}   ({p['root']})")
    print(f"train/good    : {len(train_good)} images")
    print(f"test/good     : {len(test_good)} images")
    if not train_good or not test_good or not classes:
        print("\n! folder structure incomplete. Expected:")
        print(f"  {p['root']}/train/good/*.png")
        print(f"  {p['root']}/test/good/*.png")
        print(f"  {p['root']}/test/<defect_type>/*.png")
        print(f"  {p['root']}/ground_truth/<defect_type>/*_mask.png")
        return False

    print(f"{'defect type':<22}{'images':>8}{'masks':>8}")
    for cls in classes:
        imgs = common.list_images(os.path.join(p["test"], cls))
        masks = sum(os.path.exists(common.mask_path_for(i, cfg)) for i in imgs)
        flag = "" if masks == len(imgs) else "   <- missing masks"
        if masks < len(imgs):
            ok = False
        print(f"{cls:<22}{len(imgs):>8}{masks:>8}{flag}")

    k = cfg["dataset"]["k_shot"]
    smallest = min(len(common.list_images(os.path.join(p["test"], c))) for c in classes)
    if smallest <= k:
        print(f"\n! k_shot={k} but the smallest class has {smallest} images; lower dataset.k_shot in config.yaml")
        ok = False
    return ok


def main():
    cfg = common.load_config()
    common.ensure_dirs(cfg)
    mvtec_dir = cfg["paths"]["mvtec_dir"]
    category = cfg["dataset"]["name"]
    os.makedirs(mvtec_dir, exist_ok=True)

    print("=" * 60)
    print("MVTec AD setup")
    print("=" * 60)

    if not os.path.isdir(os.path.join(mvtec_dir, category)):
        if not try_extract(mvtec_dir, category):
            print(f"\nCategory '{category}' not found under {os.path.abspath(mvtec_dir)}")
            print("Either:")
            print(f"  - place mvtec_ad.tar.xz (or {category}.tar.xz) in {mvtec_dir} and re-run, or")
            print(f"  - copy an already-extracted '{category}' folder to {os.path.join(mvtec_dir, category)}")
            print("Download (free, registration): https://www.mvtec.com/company/research/datasets/mvtec-ad")
            sys.exit(1)

    if verify(cfg):
        print("\nOK. Next: python scripts/2_make_splits_and_synthetic.py")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
