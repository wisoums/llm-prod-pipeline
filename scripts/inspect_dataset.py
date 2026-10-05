from datasets import load_dataset

ds = load_dataset(
    "ronantakizawa/github-codereview",
    revision="c3e3c6e7e9f61e3e7a5b52894bcd440d586ae6ca",
)

print(ds)

for split_name, split in ds.items():
    print(f"{split_name}: {len(split):,} rows")

python_ds = ds.filter(lambda x: x["language"] == "Python")

print("\nPython rows:")
for split_name, split in python_ds.items():
    print(f"{split_name}: {len(split):,}")

print("\nSample:")
example = python_ds["train"][0]

for key, value in example.items():
    print(f"\n--- {key} ---")
    print(value)
