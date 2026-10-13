import argparse

import fiftyone as fo

DATASET_NAME = "3d_net_rgbt"
LABEL_FIELD  = "ground_truth"
HARD_REPLACE = False
DRY_RUN      = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull CVAT annotations into a FiftyOne dataset.",
    )
    parser.add_argument(
        "--name",
        default=DATASET_NAME,
        help="FiftyOne dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--run",
        default=None,
        metavar="ANNOTATION_RUN_KEY",
        help="Pull only this annotation run key. Pulls all runs when omitted.",
    )
    parser.add_argument(
        "--filter",
        default=None,
        metavar="SUBSTRING",
        help="Pull only annotation runs whose key contains this substring (e.g. 'hausern_05_05_26').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DRY_RUN,
        help="Print which annotation runs would be pulled without loading anything.",
    )
    parser.add_argument(
        "--hard-replace",
        action="store_true",
        default=HARD_REPLACE,
        help=f"Delete the '{LABEL_FIELD}' field before loading (default: %(default)s)",
    )
    args = parser.parse_args()

    dataset = fo.load_dataset(args.name)

    if args.hard_replace:
        dataset.delete_sample_field(LABEL_FIELD)
        print(f"Deleted all labels in field: {LABEL_FIELD}")

    if args.run:
        keys = [args.run]
    else:
        keys = dataset.list_annotation_runs()
        if args.filter:
            keys = [k for k in keys if args.filter in k]
            print(f"Filtered to {len(keys)} run(s) matching '{args.filter}'")

    if args.dry_run:
        print(f"{len(keys)} run(s) would be pulled:")
        for key in keys:
            print(f"  {key}")
        return

    for key in keys:
        dataset.load_annotations(key, unexpected="prompt")
        print(f"Loaded annotations from run: {key}")

    dataset.save()


if __name__ == "__main__":
    main()
