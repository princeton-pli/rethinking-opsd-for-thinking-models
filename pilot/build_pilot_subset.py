# Build the 512-problem pilot subset from OpenThoughts-15k.
# Run from repo root on the login node (CPU, seconds).
import pandas as pd

SRC = "data/raw/openthoughts_math_filtered_15k.parquet"
DST = "data/raw/ot15k_pilot512.parquet"
N = 512

df = pd.read_parquet(SRC)
print(f"source: {len(df)} rows, columns: {list(df.columns)}")
sub = df.head(N).copy()

# MathTask.get_gold reads lowercase 'answer' first; the OT parquet carries
# capital-A 'Answer' (see GOLD_KEY=Answer in submit_experiment.sh). Mirror it
# so grading uses the clean gold instead of falling back to solution parsing.
if "answer" not in sub.columns:
    assert "Answer" in sub.columns, f"expected 'Answer' column, have {list(sub.columns)}"
    sub["answer"] = sub["Answer"].astype(str)

assert "problem" in sub.columns, "expected 'problem' column"
sub.to_parquet(DST, index=False)
print(f"wrote {len(sub)} rows -> {DST}")
