import random
import re
import statistics
from collections import Counter

from datasets import load_dataset

DATASET = "ronantakizawa/github-codereview"
DATASET_REVISION = "c3e3c6e7e9f61e3e7a5b52894bcd440d586ae6ca"
SEED = 42

ds = load_dataset(
    DATASET,
    revision=DATASET_REVISION,
)
python_ds = ds.filter(lambda x: x["language"] == "Python")

all_rows = []
for split_name, split in python_ds.items():
    for row in split:
        row["_split"] = split_name
        all_rows.append(row)

print(f"\nTotal Python examples: {len(all_rows):,}")


# --------------------------------------------------
# 1. Positive / negative distribution
# --------------------------------------------------

negatives = [r for r in all_rows if r["is_negative"]]
positives = [r for r in all_rows if not r["is_negative"]]

print("\n=== POSITIVE / NEGATIVE ===")
print(f"Positive: {len(positives):,}")
print(f"Negative: {len(negatives):,}")
print(f"Negative ratio: {len(negatives) / len(all_rows):.2%}")


# --------------------------------------------------
# 2. Quality scores
# --------------------------------------------------

scores = [float(r["quality_score"]) for r in all_rows if r["quality_score"] is not None]

print("\n=== QUALITY SCORE ===")
print(f"Mean:   {statistics.mean(scores):.3f}")
print(f"Median: {statistics.median(scores):.3f}")
print(f"Min:    {min(scores):.3f}")
print(f"Max:    {max(scores):.3f}")

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    count = sum(s < threshold for s in scores)
    print(f"Below {threshold}: {count:,} ({count / len(scores):.2%})")


# --------------------------------------------------
# 3. Comment types
# --------------------------------------------------

print("\n=== COMMENT TYPES ===")
for name, count in Counter(r["comment_type"] for r in all_rows).most_common():
    print(f"{name}: {count:,}")


# --------------------------------------------------
# 4. Vague / generic comments
# --------------------------------------------------

vague_patterns = [
    r"^lgtm[.!]*$",
    r"^looks good[.!]*$",
    r"^looks fine[.!]*$",
    r"^nice[.!]*$",
    r"^good[.!]*$",
    r"^thanks[.!]*$",
    r"^great[.!]*$",
    r"^ok[.!]*$",
]

vague = []

for row in positives:
    comment = (row["reviewer_comment"] or "").strip().lower()

    if any(re.fullmatch(pattern, comment) for pattern in vague_patterns):
        vague.append(row)

print("\n=== OBVIOUSLY VAGUE COMMENTS ===")
print(f"Found: {len(vague):,}")


# --------------------------------------------------
# 5. Missing / malformed
# --------------------------------------------------

missing_diff = [r for r in all_rows if not (r["diff_context"] or "").strip()]
missing_comment = [r for r in all_rows if not (r["reviewer_comment"] or "").strip()]

print("\n=== MISSING DATA ===")
print(f"Missing diff_context: {len(missing_diff):,}")
print(f"Missing reviewer_comment: {len(missing_comment):,}")


# --------------------------------------------------
# 6. Duplicate comments
# --------------------------------------------------

comments = [
    (r["reviewer_comment"] or "").strip()
    for r in positives
    if (r["reviewer_comment"] or "").strip()
]

counts = Counter(comments)

duplicate_instances = sum(count - 1 for count in counts.values() if count > 1)
duplicate_comment_texts = sum(1 for count in counts.values() if count > 1)

print("\n=== EXACT COMMENT DUPLICATES ===")
print(f"Repeated comment texts: {duplicate_comment_texts:,}")
print(f"Extra duplicate instances: {duplicate_instances:,}")


# --------------------------------------------------
# 7. Code size / overly long examples
# --------------------------------------------------

diff_lengths = [len(r["diff_context"] or "") for r in all_rows]
sorted_lengths = sorted(diff_lengths)

def percentile(values, p):
    index = int((len(values) - 1) * p)
    return values[index]

print("\n=== DIFF_CONTEXT LENGTH (characters) ===")
print(f"Median: {percentile(sorted_lengths, 0.50):,}")
print(f"P90:    {percentile(sorted_lengths, 0.90):,}")
print(f"P95:    {percentile(sorted_lengths, 0.95):,}")
print(f"P99:    {percentile(sorted_lengths, 0.99):,}")
print(f"Max:    {max(diff_lengths):,}")


# --------------------------------------------------
# 8. Potential external-context comments
# --------------------------------------------------

context_phrases = [
    "as discussed",
    "as mentioned",
    "previous comment",
    "previously",
    "this pr",
    "other pr",
    "other comment",
    "see above",
    "see below",
    "as we discussed",
]

context_dependent = []

for row in positives:
    comment = (row["reviewer_comment"] or "").lower()

    if any(phrase in comment for phrase in context_phrases):
        context_dependent.append(row)

print("\n=== POSSIBLE EXTERNAL-CONTEXT COMMENTS ===")
print(f"Found: {len(context_dependent):,}")


# --------------------------------------------------
# 9. Inspect negative examples
# --------------------------------------------------

neg_comments = Counter(
    (r["reviewer_comment"] or "").strip()
    for r in negatives
)

print("\n=== NEGATIVE COMMENT VALUES ===")
for value, count in neg_comments.most_common(10):
    print(repr(value), count)


# --------------------------------------------------
# 10. Random manual audit sample
# --------------------------------------------------

random.seed(SEED)

sample = random.sample(all_rows, min(100, len(all_rows)))

with open("dataset_audit_sample.txt", "w") as f:
    for i, row in enumerate(sample, 1):
        f.write("=" * 100 + "\n")
        f.write(f"EXAMPLE {i}\n")
        f.write(f"Split: {row['_split']}\n")
        f.write(f"Quality: {row['quality_score']}\n")
        f.write(f"Type: {row['comment_type']}\n")
        f.write(f"Negative: {row['is_negative']}\n")
        f.write(f"Repo: {row['repo_name']}\n\n")

        f.write("--- DIFF ---\n")
        f.write(str(row["diff_context"]) + "\n\n")

        f.write("--- REVIEW COMMENT ---\n")
        f.write(str(row["reviewer_comment"]) + "\n\n")

print("\nCreated dataset_audit_sample.txt with 100 deterministic examples.")