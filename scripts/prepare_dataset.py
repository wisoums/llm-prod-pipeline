from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

DATASET_NAME = "ronantakizawa/github-codereview"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"

SEED = 42
DEFAULT_MAX_SEQUENCE_TOKENS = 2048

SYSTEM_PROMPT = (
    "You are a Python code reviewer. Review the provided diff and return one "
    "concise, useful review comment. You may suggest a concrete change when "
    "appropriate. If there is no meaningful issue, return exactly: "
    '"No issues found."'
)

CONTEXT_DEPENDENT_PATTERNS = [
    r"\bas discussed\b",
    r"\bas mentioned\b",
    r"\bprevious comment\b",
    r"\bpreviously\b",
    r"\bsee above\b",
    r"\bsee below\b",
    r"\bas we discussed\b",
    r"\bsame as above\b",
]

STAT_KEYS = [
    "original",
    "removed_non_python",
    "removed_empty_diff",
    "removed_empty_comment",
    "removed_context_dependent",
    "removed_too_long",
    "removed_exact_duplicate",
    "kept",
    "kept_positive",
    "kept_negative",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the ReviewPy Python code-review dataset."
    )

    parser.add_argument(
        "--dataset-revision",
        required=True,
        help="Pinned Hugging Face dataset revision/commit SHA.",
    )

    parser.add_argument(
        "--tokenizer-revision",
        default=DEFAULT_TOKENIZER_REVISION,
        help="Pinned Hugging Face tokenizer revision/commit SHA.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for generated JSONL files.",
    )

    parser.add_argument(
        "--max-sequence-tokens",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_TOKENS,
        help="Maximum token count for the full chat-formatted training example.",
    )

    return parser.parse_args()


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""

    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def is_effectively_empty_diff(diff: str) -> bool:
    if not diff:
        return True

    meaningful_lines = []

    for line in diff.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith(
            ("diff --git", "index ", "--- ", "+++ ", "@@")
        ):
            continue

        meaningful_lines.append(stripped)

    return len(meaningful_lines) == 0


def is_context_dependent(comment: str) -> bool:
    lowered = comment.lower()

    return any(
        re.search(pattern, lowered)
        for pattern in CONTEXT_DEPENDENT_PATTERNS
    )


def example_key(diff: str, comment: str) -> str:
    normalized = (
        normalize_text(diff)
        + "\n<REVIEWPY-SEPARATOR>\n"
        + normalize_text(comment)
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def pull_request_key(record: dict) -> str | None:
    metadata = record["metadata"]

    repo = metadata.get("repo_name")
    pr_number = metadata.get("pr_number")

    if repo is None or pr_number is None:
        return None

    return f"{repo}#{pr_number}"


def build_messages(
    diff: str,
    comment: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": diff,
        },
        {
            "role": "assistant",
            "content": comment,
        },
    ]


def get_full_sequence_length(
    tokenizer,
    diff: str,
    comment: str,
) -> int:
    messages = build_messages(
        diff,
        comment,
    )

    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    return len(token_ids)


def build_record(
    row: dict,
    split_name: str,
) -> dict:
    diff = normalize_text(
        row.get("diff_context")
    )

    comment = normalize_text(
        row.get("reviewer_comment")
    )

    return {
        "system": SYSTEM_PROMPT,
        "user": diff,
        "assistant": comment,
        "metadata": {
            "split": split_name,
            "repo_name": row.get("repo_name"),
            "file_path": row.get("file_path"),
            "comment_line": row.get("comment_line"),
            "quality_score": row.get("quality_score"),
            "comment_type": row.get("comment_type"),
            "is_negative": row.get("is_negative"),
            "pr_number": row.get("pr_number"),
        },
    }


def write_jsonl(
    records: list[dict],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def prepare_split(
    split,
    split_name: str,
    tokenizer,
    max_sequence_tokens: int,
) -> tuple[
    list[dict],
    Counter,
    set[str],
]:
    stats = Counter(
        {
            key: 0
            for key in STAT_KEYS
        }
    )

    cleaned_records = []
    seen_examples = set()

    for row in split:
        stats["original"] += 1

        if row.get("language") != "Python":
            stats["removed_non_python"] += 1
            continue

        diff = normalize_text(
            row.get("diff_context")
        )

        comment = normalize_text(
            row.get("reviewer_comment")
        )

        if is_effectively_empty_diff(diff):
            stats["removed_empty_diff"] += 1
            continue

        if not comment:
            stats["removed_empty_comment"] += 1
            continue

        if is_context_dependent(comment):
            stats["removed_context_dependent"] += 1
            continue

        sequence_length = get_full_sequence_length(
            tokenizer,
            diff,
            comment,
        )

        if (
            sequence_length
            > max_sequence_tokens
        ):
            stats["removed_too_long"] += 1
            continue

        key = example_key(
            diff,
            comment,
        )

        if key in seen_examples:
            stats["removed_exact_duplicate"] += 1
            continue

        seen_examples.add(key)

        cleaned_records.append(
            build_record(
                row,
                split_name,
            )
        )

        stats["kept"] += 1

        if row.get("is_negative"):
            stats["kept_negative"] += 1
        else:
            stats["kept_positive"] += 1

    return (
        cleaned_records,
        stats,
        seen_examples,
    )


def recalculate_kept_stats(
    records: list[dict],
    stats: Counter,
) -> None:
    stats["kept"] = len(records)

    stats["kept_positive"] = sum(
        not record["metadata"]["is_negative"]
        for record in records
    )

    stats["kept_negative"] = sum(
        record["metadata"]["is_negative"]
        for record in records
    )


def print_stats(
    split_name: str,
    stats: Counter,
) -> None:
    print(
        f"\n=== {split_name.upper()} ==="
    )

    for key in STAT_KEYS:
        print(
            f"{key:30} "
            f"{stats[key]:,}"
        )

    print(
        f"{'removed_cross_split_duplicate':30} "
        f"{stats['removed_cross_split_duplicate']:,}"
    )

    print(
        f"{'removed_cross_split_pr':30} "
        f"{stats['removed_cross_split_pr']:,}"
    )


def main() -> None:
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    print(
        f"Dataset:  {DATASET_NAME}"
    )

    print(
        f"Revision: {args.dataset_revision}"
    )

    print(
        f"Model:    {MODEL_NAME}"
    )

    print(
        "Tokenizer revision: "
        f"{args.tokenizer_revision}"
    )

    print(
        f"Seed:     {SEED}"
    )

    print(
        "Max sequence tokens: "
        f"{args.max_sequence_tokens}"
    )

    print("\nLoading tokenizer...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=args.tokenizer_revision,
        )
    )

    print("Loading dataset...")

    dataset: DatasetDict = load_dataset(
        DATASET_NAME,
        revision=args.dataset_revision,
    )

    manifest = {
        "dataset": DATASET_NAME,
        "dataset_revision": (
            args.dataset_revision
        ),
        "model": MODEL_NAME,
        "tokenizer_revision": (
            args.tokenizer_revision
        ),
        "seed": SEED,
        "max_sequence_tokens": (
            args.max_sequence_tokens
        ),
        "system_prompt": SYSTEM_PROMPT,
        "splits": {},
        "cross_split_overlap": {},
    }

    cleaned_by_split: dict[
        str,
        list[dict],
    ] = {}

    keys_by_split: dict[
        str,
        set[str],
    ] = {}

    stats_by_split: dict[
        str,
        Counter,
    ] = {}

    # --------------------------------------------------
    # 1. Clean each split independently
    # --------------------------------------------------

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        if split_name not in dataset:
            raise ValueError(
                "Missing expected split: "
                f"{split_name}"
            )

        (
            cleaned_records,
            stats,
            keys,
        ) = prepare_split(
            dataset[split_name],
            split_name=split_name,
            tokenizer=tokenizer,
            max_sequence_tokens=(
                args.max_sequence_tokens
            ),
        )

        cleaned_by_split[
            split_name
        ] = cleaned_records

        keys_by_split[
            split_name
        ] = keys

        stats_by_split[
            split_name
        ] = stats

    # --------------------------------------------------
    # 2. Remove exact cross-split duplicate examples
    # --------------------------------------------------

    train_keys = keys_by_split["train"]

    validation_records = []
    validation_keys = set()

    removed_validation_overlap = 0

    for record in cleaned_by_split[
        "validation"
    ]:
        key = example_key(
            record["user"],
            record["assistant"],
        )

        if key in train_keys:
            removed_validation_overlap += 1
            continue

        validation_records.append(
            record
        )

        validation_keys.add(key)

    cleaned_by_split[
        "validation"
    ] = validation_records

    keys_by_split[
        "validation"
    ] = validation_keys

    test_records = []
    test_keys = set()

    removed_test_overlap = 0

    blocked_test_keys = (
        train_keys
        | validation_keys
    )

    for record in cleaned_by_split[
        "test"
    ]:
        key = example_key(
            record["user"],
            record["assistant"],
        )

        if key in blocked_test_keys:
            removed_test_overlap += 1
            continue

        test_records.append(
            record
        )

        test_keys.add(key)

    cleaned_by_split[
        "test"
    ] = test_records

    keys_by_split[
        "test"
    ] = test_keys

    stats_by_split[
        "train"
    ][
        "removed_cross_split_duplicate"
    ] = 0

    stats_by_split[
        "validation"
    ][
        "removed_cross_split_duplicate"
    ] = removed_validation_overlap

    stats_by_split[
        "test"
    ][
        "removed_cross_split_duplicate"
    ] = removed_test_overlap

    # --------------------------------------------------
    # 3. Remove cross-split PR leakage
    # --------------------------------------------------

    train_pr_keys = {
        key
        for record in cleaned_by_split[
            "train"
        ]
        if (
            key := pull_request_key(
                record
            )
        )
        is not None
    }

    validation_records = []
    validation_pr_keys = set()

    removed_validation_pr_overlap = 0

    for record in cleaned_by_split[
        "validation"
    ]:
        key = pull_request_key(
            record
        )

        if (
            key is not None
            and key in train_pr_keys
        ):
            removed_validation_pr_overlap += 1
            continue

        validation_records.append(
            record
        )

        if key is not None:
            validation_pr_keys.add(key)

    cleaned_by_split[
        "validation"
    ] = validation_records

    blocked_test_pr_keys = (
        train_pr_keys
        | validation_pr_keys
    )

    test_records = []

    removed_test_pr_overlap = 0

    for record in cleaned_by_split[
        "test"
    ]:
        key = pull_request_key(
            record
        )

        if (
            key is not None
            and key in blocked_test_pr_keys
        ):
            removed_test_pr_overlap += 1
            continue

        test_records.append(
            record
        )

    cleaned_by_split[
        "test"
    ] = test_records

    stats_by_split[
        "train"
    ][
        "removed_cross_split_pr"
    ] = 0

    stats_by_split[
        "validation"
    ][
        "removed_cross_split_pr"
    ] = removed_validation_pr_overlap

    stats_by_split[
        "test"
    ][
        "removed_cross_split_pr"
    ] = removed_test_pr_overlap

    # Recalculate counts after leakage removal
    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        recalculate_kept_stats(
            cleaned_by_split[
                split_name
            ],
            stats_by_split[
                split_name
            ],
        )

    # Rebuild exact keys after all filtering
    keys_by_split = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        keys_by_split[
            split_name
        ] = {
            example_key(
                record["user"],
                record["assistant"],
            )
            for record in cleaned_by_split[
                split_name
            ]
        }

    # --------------------------------------------------
    # 4. Save cleaned splits
    # --------------------------------------------------

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        output_path = (
            output_dir
            / f"{split_name}.jsonl"
        )

        write_jsonl(
            cleaned_by_split[
                split_name
            ],
            output_path,
        )

        stats = stats_by_split[
            split_name
        ]

        manifest["splits"][
            split_name
        ] = {
            **{
                key: stats[key]
                for key in STAT_KEYS
            },
            "removed_cross_split_duplicate": (
                stats[
                    "removed_cross_split_duplicate"
                ]
            ),
            "removed_cross_split_pr": (
                stats[
                    "removed_cross_split_pr"
                ]
            ),
            "output_file": str(
                output_path
            ),
            "sha256": sha256_file(
                output_path
            ),
        }

        print_stats(
            split_name,
            stats,
        )

    # --------------------------------------------------
    # 5. Verify no exact overlap remains
    # --------------------------------------------------

    overlap_pairs = [
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

    print(
        "\n=== CROSS-SPLIT EXACT OVERLAP ==="
    )

    for left, right in overlap_pairs:
        overlap = len(
            keys_by_split[left]
            & keys_by_split[right]
        )

        label = (
            f"{left}_vs_{right}"
        )

        manifest[
            "cross_split_overlap"
        ][label] = overlap

        print(
            f"{label:30} "
            f"{overlap:,}"
        )

    # --------------------------------------------------
    # 6. Save manifest
    # --------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== DONE ===")

    print(
        "Processed dataset written to: "
        f"{output_dir}"
    )

    print(
        "Manifest written to:          "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()