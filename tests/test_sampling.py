import pytest
import pandas as pd
from utils.sampling import stratified_sample

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature": range(10),
        "class": ["A"]*5 + ["B"]*5
    })

def test_stratified_sample_fraction(sample_df):
    # Sample 60% of each class
    result = stratified_sample(sample_df, stratify_col="class", sample_size=0.6, random_state=1)
    # Should have 3 from each class (since int(5*0.6) == 3)
    counts = result["class"].value_counts()
    assert counts["A"] == 3
    assert counts["B"] == 3
    # Should not have any duplicates
    assert result.duplicated().sum() == 0

def test_stratified_sample_count(sample_df):
    # Sample 2 from each class
    result = stratified_sample(sample_df, stratify_col="class", sample_size=2, random_state=1)
    counts = result["class"].value_counts()
    assert counts["A"] == 2
    assert counts["B"] == 2

def test_fraction_out_of_bounds(sample_df):
    with pytest.raises(ValueError):
        stratified_sample(sample_df, stratify_col="class", sample_size=1.5)
    with pytest.raises(ValueError):
        stratified_sample(sample_df, stratify_col="class", sample_size=0)

def test_group_smaller_than_n(sample_df):
    # Add a small group
    df = pd.concat([sample_df, pd.DataFrame({"feature": [100], "class": ["C"]})], ignore_index=True)
    # Request 2 samples per group, but "C" only has 1 row
    result = stratified_sample(df, stratify_col="class", sample_size=2, random_state=1)
    counts = result["class"].value_counts()
    assert counts["C"] == 1
    assert counts["A"] == 2
    assert counts["B"] == 2

def test_nonexistent_stratify_column(sample_df):
    with pytest.raises(KeyError):
        stratified_sample(sample_df, stratify_col="not_a_column", sample_size=0.5)

def test_empty_dataframe():
    df = pd.DataFrame(columns=["feature", "class"])
    result = stratified_sample(df, stratify_col="class", sample_size=0.5)
    assert result.empty

def test_random_state_reproducibility(sample_df):
    result1 = stratified_sample(sample_df, stratify_col="class", sample_size=0.5, random_state=42)
    result2 = stratified_sample(sample_df, stratify_col="class", sample_size=0.5, random_state=42)
    pd.testing.assert_frame_equal(result1.reset_index(drop=True), result2.reset_index(drop=True))