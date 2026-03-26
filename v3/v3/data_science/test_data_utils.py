import pytest
import pandas as pd

def clean_data(df):
    """Remove rows with missing values and duplicates."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def test_clean_data():
    data = {'A': [1, 2, None, 2], 'B': [5, 6, 7, 6]}
    df = pd.DataFrame(data)
    cleaned = clean_data(df)
    assert len(cleaned) == 1  # only row 1 remains
    assert cleaned.iloc[0]['A'] == 2
    assert cleaned.iloc[0]['B'] == 6

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
