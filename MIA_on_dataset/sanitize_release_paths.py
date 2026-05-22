#!/usr/bin/env python3
"""Replace repo-local absolute paths in release text artifacts."""

import argparse
from pathlib import Path


TEXT_SUFFIXES = {".csv", ".json", ".txt"}


def sanitize_text_files(root, old_prefix, new_prefix, dry_run=False):
    changed = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        rewritten = text.replace(old_prefix, new_prefix)
        if rewritten == text:
            continue
        changed.append(path)
        if not dry_run:
            path.write_text(rewritten, encoding="utf-8")
    return changed


def main():
    mia_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=mia_root / "two_stage_mia_results",
        help="Directory containing release text artifacts.",
    )
    parser.add_argument(
        "--old-prefix",
        default=str(mia_root) + "/",
        help="Absolute prefix to replace.",
    )
    parser.add_argument(
        "--new-prefix",
        default="MIA_on_dataset/",
        help="Portable prefix written into release artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changed files without writing them.")
    args = parser.parse_args()

    changed = sanitize_text_files(args.root, args.old_prefix, args.new_prefix, dry_run=args.dry_run)
    action = "Would sanitize" if args.dry_run else "Sanitized"
    print("{} {} text files under {}".format(action, len(changed), args.root))


if __name__ == "__main__":
    main()
