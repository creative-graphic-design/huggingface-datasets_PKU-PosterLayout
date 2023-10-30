import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "PKU-PosterLayout.py"


@pytest.mark.parametrize(
    argnames=(
        "expected_num_train",
        "expected_num_test",
    ),
    argvalues=((9974, 905),),
)
def test_load_dataset(dataset_path: str, expected_num_train: int, expected_num_test):
    dataset = ds.load_dataset(path=dataset_path, token=True)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["test"].num_rows == expected_num_test
