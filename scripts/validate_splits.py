"""
Validate the processed dataset splits.

By default this script is read-only: it fails (exit status 1) if any
cross-split overlap is found, or if the current splits no longer match
the frozen manifest in data/metadata/split_manifest.json.

Regenerating the manifest is a separate, explicit operation:

    python scripts/validate_splits.py --write-manifest

which still refuses to write when any overlap remains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROCESSED_DIR = Path(
    "data/processed"
)

MANIFEST_PATH = Path(
    "data/metadata/split_manifest.json"
)

SPLITS = [
    "train",
    "validation",
    "test",
]

MIN_NORMALIZED_DIFF_CHARS = 80


def normalize_text(
    text: str,
) -> str:
    return (
        text.replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .strip()
    )


def normalize_diff_for_similarity(
    diff: str,
) -> str:
    """
    Remove formatting details that do not strongly affect the meaning
    of the code change.

    This is a lightweight near-duplicate check.
    """

    lines = []

    for line in normalize_text(
        diff
    ).splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith(
            (
                "diff --git",
                "index ",
                "--- ",
                "+++ ",
                "@@",
            )
        ):
            continue

        stripped = re.sub(
            r"\s+",
            " ",
            stripped,
        )

        lines.append(
            stripped
        )

    return "\n".join(
        lines
    )


def sha256_text(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_jsonl(
    path: Path,
) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            records.append(
                json.loads(
                    line
                )
            )

    return records


def record_key(
    record: dict,
) -> str:
    """
    Same diff + same review comment.
    """

    combined = (
        normalize_text(
            record["user"]
        )
        + "\n<REVIEWPY-SEPARATOR>\n"
        + normalize_text(
            record["assistant"]
        )
    )

    return sha256_text(
        combined
    )


def diff_key(
    record: dict,
) -> str:
    """
    Same raw input diff,
    regardless of review comment.
    """

    return sha256_text(
        normalize_text(
            record["user"]
        )
    )


def normalized_diff_key(
    record: dict,
) -> str | None:
    """
    Lightweight near-duplicate check.

    Tiny matches such as "+import logging" are ignored because
    they are not meaningful evidence of dataset leakage.
    """

    normalized = (
        normalize_diff_for_similarity(
            record["user"]
        )
    )

    if (
        len(normalized)
        < MIN_NORMALIZED_DIFF_CHARS
    ):
        return None

    return sha256_text(
        normalized
    )


def pr_key(
    record: dict,
) -> str | None:
    metadata = record[
        "metadata"
    ]

    repo = metadata.get(
        "repo_name"
    )

    pr_number = metadata.get(
        "pr_number"
    )

    if (
        repo is None
        or pr_number is None
    ):
        return None

    return (
        f"{repo}#{pr_number}"
    )


def overlap_count(
    left: set[str],
    right: set[str],
) -> int:
    return len(
        left & right
    )


def summarize_split(
    records: list[dict],
) -> dict:
    comment_types = Counter()

    positive = 0
    negative = 0

    for record in records:
        metadata = record[
            "metadata"
        ]

        comment_type = (
            metadata.get(
                "comment_type"
            )
        )

        if comment_type:
            comment_types[
                comment_type
            ] += 1

        if metadata.get(
            "is_negative"
        ):
            negative += 1
        else:
            positive += 1

    total = len(records)

    return {
        "count": total,
        "positive": positive,
        "negative": negative,
        "negative_ratio": (
            round(
                negative / total,
                4,
            )
            if total
            else 0
        ),
        "comment_types": dict(
            comment_types
        ),
    }


def build_index(
    records: list[dict],
    key_function,
) -> dict[str, list[dict]]:
    index: dict[
        str,
        list[dict],
    ] = {}

    for record in records:
        key = key_function(
            record
        )

        if key is None:
            continue

        index.setdefault(
            key,
            [],
        ).append(
            record
        )

    return index


def find_overlaps(
    overlap_report: dict[str, dict[str, int]],
) -> list[str]:
    """
    Every nonzero overlap metric, as "pair.metric=count".
    """

    return [
        f"{label}.{metric}={count}"
        for (
            label,
            metrics,
        ) in overlap_report.items()
        for (
            metric,
            count,
        ) in metrics.items()
        if count
    ]


def flatten(
    value,
    prefix: str = "",
) -> dict[str, object]:
    if not isinstance(
        value,
        dict,
    ):
        return {
            prefix: value
        }

    flat: dict[
        str,
        object,
    ] = {}

    for key, child in value.items():
        flat.update(
            flatten(
                child,
                (
                    f"{prefix}.{key}"
                    if prefix
                    else key
                ),
            )
        )

    return flat


def diff_manifests(
    expected: dict,
    actual: dict,
) -> list[str]:
    """
    Every field that differs between the frozen manifest
    and the manifest computed from the current splits.
    """

    expected_flat = flatten(
        expected
    )

    actual_flat = flatten(
        actual
    )

    missing = object()

    return [
        f"{key}: expected "
        f"{expected_flat.get(key, '<missing>')!r}, "
        "got "
        f"{actual_flat.get(key, '<missing>')!r}"
        for key in sorted(
            expected_flat.keys()
            | actual_flat.keys()
        )
        if expected_flat.get(
            key,
            missing,
        )
        != actual_flat.get(
            key,
            missing,
        )
    ]


def parse_args(
    argv: list[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check processed splits for cross-split leakage "
            "and verify them against the frozen manifest."
        ),
    )

    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=(
            "Regenerate the frozen manifest from the current "
            "splits instead of verifying against it. Refuses "
            "to write if any overlap remains."
        ),
    )

    return parser.parse_args(
        argv
    )


def main(
    argv: list[str] | None = None,
) -> int:
    args = parse_args(
        argv
    )

    records_by_split: dict[
        str,
        list[dict],
    ] = {}

    # --------------------------------------------------
    # 1. Load cleaned dataset
    # --------------------------------------------------

    for split in SPLITS:
        path = (
            PROCESSED_DIR
            / f"{split}.jsonl"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. "
                "Run scripts/prepare_dataset.py first."
            )

        records_by_split[
            split
        ] = load_jsonl(
            path
        )

    # --------------------------------------------------
    # 2. Build leakage-check keys
    # --------------------------------------------------

    exact_keys = {}
    diff_keys = {}
    normalized_diff_keys = {}
    pr_keys = {}

    for (
        split,
        records,
    ) in records_by_split.items():

        exact_keys[
            split
        ] = {
            record_key(
                record
            )
            for record in records
        }

        diff_keys[
            split
        ] = {
            diff_key(
                record
            )
            for record in records
        }

        normalized_diff_keys[
            split
        ] = {
            key
            for record in records
            if (
                key := normalized_diff_key(
                    record
                )
            )
            is not None
        }

        pr_keys[
            split
        ] = {
            key
            for record in records
            if (
                key := pr_key(
                    record
                )
            )
            is not None
        }

    normalized_diff_indexes = {
        split: build_index(
            records,
            normalized_diff_key,
        )
        for (
            split,
            records,
        ) in records_by_split.items()
    }

    pr_indexes = {
        split: build_index(
            records,
            pr_key,
        )
        for (
            split,
            records,
        ) in records_by_split.items()
    }

    pairs = [
        (
            "train",
            "validation",
        ),
        (
            "train",
            "test",
        ),
        (
            "validation",
            "test",
        ),
    ]

    overlap_report = {}

    # --------------------------------------------------
    # 3. Leakage summary
    # --------------------------------------------------

    print(
        "\n=== SPLIT LEAKAGE CHECKS ==="
    )

    for left, right in pairs:
        label = (
            f"{left}_vs_{right}"
        )

        exact_overlap = (
            overlap_count(
                exact_keys[left],
                exact_keys[right],
            )
        )

        diff_overlap = (
            overlap_count(
                diff_keys[left],
                diff_keys[right],
            )
        )

        normalized_diff_overlap = (
            overlap_count(
                normalized_diff_keys[
                    left
                ],
                normalized_diff_keys[
                    right
                ],
            )
        )

        pr_overlap = (
            overlap_count(
                pr_keys[left],
                pr_keys[right],
            )
        )

        overlap_report[
            label
        ] = {
            "exact_record_overlap": (
                exact_overlap
            ),
            "exact_diff_overlap": (
                diff_overlap
            ),
            "normalized_diff_overlap": (
                normalized_diff_overlap
            ),
            "pr_overlap": (
                pr_overlap
            ),
        }

        print(
            f"\n{label}"
        )

        print(
            "  exact record overlap:      "
            f"{exact_overlap:,}"
        )

        print(
            "  exact diff overlap:        "
            f"{diff_overlap:,}"
        )

        print(
            "  normalized diff overlap:   "
            f"{normalized_diff_overlap:,}"
        )

        print(
            "  PR overlap:                "
            f"{pr_overlap:,}"
        )

    # --------------------------------------------------
    # 4. Show meaningful normalized-diff overlaps
    # --------------------------------------------------

    print(
        "\n=== NORMALIZED DIFF OVERLAP DETAILS ==="
    )

    found_normalized_overlap = False

    for left, right in pairs:
        shared_keys = (
            normalized_diff_keys[
                left
            ]
            & normalized_diff_keys[
                right
            ]
        )

        if not shared_keys:
            continue

        found_normalized_overlap = True

        print(
            f"\n{left} vs {right}"
        )

        for key in shared_keys:
            print(
                "\n--- SHARED NORMALIZED DIFF ---"
            )

            for split_name in [
                left,
                right,
            ]:
                print(
                    f"\n[{split_name.upper()}]"
                )

                for record in (
                    normalized_diff_indexes[
                        split_name
                    ][key]
                ):
                    metadata = (
                        record[
                            "metadata"
                        ]
                    )

                    print(
                        "Repo: "
                        f"{metadata.get('repo_name')} "
                        "PR: "
                        f"{metadata.get('pr_number')} "
                        "File: "
                        f"{metadata.get('file_path')}"
                    )

                    print(
                        "Diff:"
                    )

                    print(
                        record[
                            "user"
                        ][:1000]
                    )

                    print(
                        "Review:"
                    )

                    print(
                        record[
                            "assistant"
                        ][:500]
                    )

    if not found_normalized_overlap:
        print(
            "No meaningful normalized "
            "diff overlaps found."
        )

    # --------------------------------------------------
    # 5. Show PR overlaps
    # --------------------------------------------------

    print(
        "\n=== PR OVERLAP DETAILS ==="
    )

    found_pr_overlap = False

    for left, right in pairs:
        shared_prs = (
            pr_keys[left]
            & pr_keys[right]
        )

        if not shared_prs:
            continue

        found_pr_overlap = True

        print(
            f"\n{left} vs {right}"
        )

        for key in shared_prs:
            print(
                f"\nShared PR: {key}"
            )

            for split_name in [
                left,
                right,
            ]:
                records = (
                    pr_indexes[
                        split_name
                    ][key]
                )

                print(
                    f"  {split_name}: "
                    f"{len(records)} example(s)"
                )

                for record in records[
                    :5
                ]:
                    metadata = (
                        record[
                            "metadata"
                        ]
                    )

                    print(
                        "    "
                        f"{metadata.get('file_path')} "
                        "line="
                        f"{metadata.get('comment_line')}"
                    )

    if not found_pr_overlap:
        print(
            "No cross-split PR overlaps found."
        )

    # --------------------------------------------------
    # 6. Build frozen split manifest
    # --------------------------------------------------

    manifest = {
        "splits": {},
        "overlap": overlap_report,
        "near_duplicate_policy": {
            "method": (
                "normalized exact match"
            ),
            "minimum_normalized_diff_chars": (
                MIN_NORMALIZED_DIFF_CHARS
            ),
        },
        "test_split_frozen": True,
    }

    print(
        "\n=== SPLIT SUMMARY ==="
    )

    for split in SPLITS:
        records = (
            records_by_split[
                split
            ]
        )

        path = (
            PROCESSED_DIR
            / f"{split}.jsonl"
        )

        summary = (
            summarize_split(
                records
            )
        )

        manifest[
            "splits"
        ][split] = {
            **summary,
            "sha256": sha256_file(
                path
            ),
        }

        print(
            f"\n{split}"
        )

        print(
            "  count:          "
            f"{summary['count']:,}"
        )

        print(
            "  positive:       "
            f"{summary['positive']:,}"
        )

        print(
            "  negative:       "
            f"{summary['negative']:,}"
        )

        print(
            "  negative ratio: "
            f"{summary['negative_ratio']:.2%}"
        )

    # --------------------------------------------------
    # 7. Fail on any cross-split leakage
    # --------------------------------------------------

    overlaps = find_overlaps(
        overlap_report
    )

    if overlaps:
        print(
            "\nFAILED: cross-split overlap detected:",
            file=sys.stderr,
        )

        for overlap in overlaps:
            print(
                f"  {overlap}",
                file=sys.stderr,
            )

        if args.write_manifest:
            print(
                "Manifest not written.",
                file=sys.stderr,
            )

        return 1

    # --------------------------------------------------
    # 8a. Explicit regeneration of the frozen manifest
    # --------------------------------------------------

    if args.write_manifest:
        MANIFEST_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with MANIFEST_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\nManifest written to: "
            f"{MANIFEST_PATH}"
        )

        return 0

    # --------------------------------------------------
    # 8b. Verify against the frozen manifest
    # --------------------------------------------------

    if not MANIFEST_PATH.exists():
        print(
            f"\nFAILED: {MANIFEST_PATH} does not exist. "
            "Run with --write-manifest to create it.",
            file=sys.stderr,
        )

        return 1

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        frozen_manifest = json.load(
            file
        )

    differences = diff_manifests(
        frozen_manifest,
        manifest,
    )

    if differences:
        print(
            "\nFAILED: splits do not match the frozen "
            f"manifest {MANIFEST_PATH}:",
            file=sys.stderr,
        )

        for difference in differences:
            print(
                f"  {difference}",
                file=sys.stderr,
            )

        print(
            "If this change is intentional, rerun with "
            "--write-manifest and commit the new manifest.",
            file=sys.stderr,
        )

        return 1

    print(
        "\nOK: no cross-split overlap and splits match "
        f"{MANIFEST_PATH}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )