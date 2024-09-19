import os

import datasets as ds
import pytest


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "PKU-PosterLayout"


@pytest.fixture
def dataset_path(dataset_name: str) -> str:
    return f"{dataset_name}.py"


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames="subset_name",
    argvalues=(
        "default",
        "ralf",
    ),
)
@pytest.mark.parametrize(
    argnames=(
        "expected_num_train",
        "expected_num_test",
    ),
    argvalues=((9974, 905),),
)
def test_load_dataset(
    dataset_path: str,
    subset_name: str,
    expected_num_train: int,
    expected_num_test,
):
    dataset = ds.load_dataset(path=dataset_path, name=subset_name, token=True)
    assert isinstance(dataset, ds.DatasetDict)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["test"].num_rows == expected_num_test


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames="subset_name",
    argvalues=(
        "default",
        "ralf",
    ),
)
def test_push_to_hub(
    repo_id: str,
    subset_name: str,
    dataset_path: str,
    seed: int = 19950815,
):
    dataset = ds.load_dataset(path=dataset_path, name=subset_name, token=True)
    assert isinstance(dataset, ds.DatasetDict)

    if subset_name == "ralf":
        #
        # Rename `test` (with no annotation) to `no_annotation`
        #
        no_annotation_dataset = dataset["test"]

        #
        # Split the dataset into train:valid:test = 8:1:1
        #
        # First, split train into train and test at 8:2
        tng_tst_set = dataset["train"].train_test_split(test_size=0.2, seed=seed)
        # Then, split test into valid and test at 1:1 ratio to make train:valid:test = 8:1:1
        val_tst_set = tng_tst_set["test"].train_test_split(test_size=0.5, seed=seed)

        # Reorganize the split dataset
        tng_dataset = tng_tst_set["train"]
        val_dataset = val_tst_set["train"]
        tst_dataset = val_tst_set["test"]

        dataset = ds.DatasetDict(
            train=tng_dataset,
            validation=val_dataset,
            test=tst_dataset,
            no_annotation=no_annotation_dataset,
        )

        # Check if the split is correct
        assert (
            tng_dataset.num_rows == 7787
            and val_dataset.num_rows == 973
            and tst_dataset.num_rows == 974
            and no_annotation_dataset.num_rows == 905
        ), dataset

    #
    # Push the dataset to the huggingface hub
    #
    dataset.push_to_hub(repo_id=repo_id, config_name=subset_name, private=True)
