import ast
import pathlib
from typing import List, Optional, TypedDict, Union, cast

import datasets as ds
import pandas as pd
from datasets.utils.logging import get_logger
from PIL import Image
from PIL.Image import Image as PilImage

logger = get_logger(__name__)

_DESCRIPTION = (
    "A New Dataset and Benchmark for Content-aware Visual-Textual Presentation Layout"
)

_CITATION = """\
@inproceedings{hsu2023posterlayout,
  title={PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout},
  author={Hsu, Hsiao Yuan and He, Xiangteng and Peng, Yuxin and Kong, Hao and Zhang, Qing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6018--6026},
  year={2023}
}
"""

_HOMEPAGE = "http://59.108.48.34/tiki/PosterLayout/"

_LICENSE = "Images in PKU PosterLayout are distributed under the CC BY-SA 4.0 license."


class TrainPoster(TypedDict):
    original: str
    inpainted: str


class TestPoster(TypedDict):
    canvas: str


class SaliencyMaps(TypedDict):
    pfpn: str
    basnet: str


class TrainDataset(TypedDict):
    poster: TrainPoster
    saliency_maps: SaliencyMaps


class TestDataset(TypedDict):
    poster: TestPoster
    saliency_maps: SaliencyMaps


class Annotation(TypedDict):
    train: str


class DatasetUrls(TypedDict):
    train: TrainDataset
    test: TestDataset
    annotation: Annotation


# The author of this loading script has uploaded the poster image and saliency maps to the HuggingFace's private repository to facilitate testing.
# If you are using this loading script, please download the annotations from the appropriate channels, such as the OneDrive link provided by the Magazine dataset's author.
# (To the author of Magazine dataset, if there are any issues regarding this matter, please contact us. We will address it promptly.)
_URLS: DatasetUrls = {
    "train": {
        "poster": {
            "original": "https://huggingface.co/datasets/shunk031-private/PKU-PosterLayout-private/resolve/main/train/original_poster.zip",
            "inpainted": "https://huggingface.co/datasets/shunk031-private/PKU-PosterLayout-private/resolve/main/train/inpainted_poster.zip",
        },
        "saliency_maps": {
            "pfpn": "https://huggingface.co/datasets/shunk031-private/PKU-PosterLayout-private/resolve/main/train/saliencymaps_pfpn.zip",
            "basnet": "https://huggingface.co/datasets/shunk031-private/PKU-PosterLayout-private/resolve/main/train/saliencymaps_basnet.zip",
        },
    },
    "test": {
        "poster": {
            "canvas": "https://huggingface.co/datasets/shunk031-private/PKU-PosterLayout-private/resolve/main/test/image_canvas.zip",
        },
        "saliency_maps": {
            "pfpn": "https://huggingface.co/datasets/shunk031-private/PKU-PosterLayout-private/resolve/main/test/saliencymaps_pfpn.zip",
            "basnet": "https://huggingface.co/datasets/shunk031-private/PKU-PosterLayout-private/resolve/main/test/saliencymaps_basnet.zip",
        },
    },
    "annotation": {
        "train": "https://huggingface.co/datasets/shunk031-private/PKU-PosterLayout-private/raw/main/annotations/train_csv_9973.csv",
    },
}


def file_sorter(f: pathlib.Path) -> int:
    idx, *_ = f.stem.split("_")
    return int(idx)


def load_image(file_path: pathlib.Path) -> PilImage:
    logger.info(f"Load from {file_path}")
    return Image.open(file_path)


def get_original_poster_files(base_dir: str) -> List[pathlib.Path]:
    poster_dir = pathlib.Path(base_dir) / "original_poster"
    return sorted(poster_dir.iterdir(), key=lambda f: int(f.stem))


def get_inpainted_poster_files(base_dir: str) -> List[pathlib.Path]:
    inpainted_dir = pathlib.Path(base_dir) / "inpainted_poster"
    return sorted(inpainted_dir.iterdir(), key=file_sorter)


def get_basnet_map_files(base_dir: str) -> List[pathlib.Path]:
    basnet_map_dir = pathlib.Path(base_dir) / "saliencymaps_basnet"
    return sorted(basnet_map_dir.iterdir(), key=file_sorter)


def get_pfpn_map_files(base_dir: str) -> List[pathlib.Path]:
    pfpn_map_dir = pathlib.Path(base_dir) / "saliencymaps_pfpn"
    return sorted(pfpn_map_dir.iterdir(), key=file_sorter)


def get_canvas_files(base_dir: str) -> List[pathlib.Path]:
    canvas_dir = pathlib.Path(base_dir) / "image_canvas"
    return sorted(canvas_dir.iterdir(), key=lambda f: int(f.stem))


class PosterLayoutDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIGS = [ds.BuilderConfig(version=VERSION)]

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "original_poster": ds.Image(),
                "inpainted_poster": ds.Image(),
                "basnet_saliency_map": ds.Image(),
                "pfpn_saliency_map": ds.Image(),
                "canvas": ds.Image(),
                "annotations": ds.Sequence(
                    {
                        "poster_path": ds.Value("string"),
                        "total_elem": ds.Value("int32"),
                        "cls_elem": ds.Value("int32"),
                        "box_elem": ds.Sequence(ds.Value("int32")),
                    }
                ),
            }
        )
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        file_paths = dl_manager.download_and_extract(_URLS)

        tng_files = file_paths["train"]  # type: ignore
        tst_files = file_paths["test"]  # type: ignore
        ann_file = file_paths["annotation"]  # type: ignore

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "poster": tng_files["poster"],
                    "saliency_maps": tng_files["saliency_maps"],
                    "annotation": ann_file["train"],
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={
                    "poster": tst_files["poster"],
                    "saliency_maps": tst_files["saliency_maps"],
                },
            ),
        ]

    def _generate_train_examples(
        self,
        poster: TrainPoster,
        saliency_maps: SaliencyMaps,
        annotation: str,
    ):
        ann_df = pd.read_csv(annotation)
        ann_df = ann_df.assign(box_elem=ann_df["box_elem"].apply(ast.literal_eval))

        poster_files = get_original_poster_files(base_dir=poster["original"])
        inpainted_files = get_inpainted_poster_files(base_dir=poster["inpainted"])

        basnet_map_files = get_basnet_map_files(base_dir=saliency_maps["basnet"])
        pfpn_map_files = get_pfpn_map_files(base_dir=saliency_maps["pfpn"])

        assert (
            len(poster_files)
            == len(inpainted_files)
            == len(basnet_map_files)
            == len(pfpn_map_files)
        )

        it = zip(poster_files, inpainted_files, basnet_map_files, pfpn_map_files)
        for i, (
            original_poster_path,
            inpainted_poster_path,
            basnet_map_path,
            pfpn_map_path,
        ) in enumerate(it):

            poster_path = f"train/{original_poster_path.name}"
            poster_anns = ann_df[ann_df["poster_path"] == poster_path]

            annotations = poster_anns.to_dict(orient="records")

            yield i, {
                "original_poster": load_image(original_poster_path),
                "inpainted_poster": load_image(inpainted_poster_path),
                "basnet_saliency_map": load_image(basnet_map_path),
                "pfpn_saliency_map": load_image(pfpn_map_path),
                "canvas": None,
                "annotations": annotations,
            }

    def _generate_test_examples(self, poster: TestPoster, saliency_maps: SaliencyMaps):
        canvas_files = get_canvas_files(base_dir=poster["canvas"])

        basnet_map_files = get_basnet_map_files(base_dir=saliency_maps["basnet"])
        pfpn_map_files = get_pfpn_map_files(base_dir=saliency_maps["pfpn"])

        assert len(canvas_files) == len(basnet_map_files) == len(pfpn_map_files)
        it = zip(canvas_files, basnet_map_files, pfpn_map_files)
        for i, (canvas_path, basnet_map_path, pfpn_map_path) in enumerate(it):
            yield i, {
                "original_poster": None,
                "inpainted_poster": None,
                "basnet_saliency_map": load_image(basnet_map_path),
                "pfpn_saliency_map": load_image(pfpn_map_path),
                "canvas": load_image(canvas_path),
                "annotations": None,
            }

    def _generate_examples(
        self,
        poster: Union[TrainPoster, TestPoster],
        saliency_maps: SaliencyMaps,
        annotation: Optional[str] = None,
    ):
        if "original" in poster and "inpainted" in poster:
            assert annotation is not None

            yield from self._generate_train_examples(
                poster=cast(TrainPoster, poster),
                saliency_maps=saliency_maps,
                annotation=annotation,
            )
        elif "canvas" in poster:
            yield from self._generate_test_examples(
                poster=cast(TestPoster, poster),
                saliency_maps=saliency_maps,
            )
        else:
            raise ValueError("Invalid dataset")
