import pandas as pd
import pytest

from src.preprocessing import get_splits, read_dataframe


@pytest.fixture
def sample_csv(tmp_path):
    data = """feature1,feature2,target
1,10,0
2,20,1
1,10,0
3,30,1
"""
    file = tmp_path / "sample.csv"
    file.write_text(data)
    return str(file)


def test_read_dataframe_removes_duplicates(sample_csv):
    df = read_dataframe(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3  # one duplicate removed
    assert df.duplicated().sum() == 0


def test_get_splits_shapes():
    df = pd.DataFrame(
        {"feature1": range(100), "feature2": range(100, 200), "target": [0, 1] * 50}
    )
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(
        df, train_size=0.8, test_size=0.5, random_state=42
    )

    assert len(X_train) + len(X_val) + len(X_test) == 100
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_get_splits_stratified():
    df = pd.DataFrame({"feature": list(range(100)), "target": [0] * 50 + [1] * 50})
    _, _, _, y_train, y_val, y_test = get_splits(
        df, train_size=0.8, test_size=0.5, random_state=42
    )

    # Check that classes are roughly balanced
    for split in [y_train, y_val, y_test]:
        assert abs(split.value_counts(normalize=True)[0] - 0.5) < 0.2
