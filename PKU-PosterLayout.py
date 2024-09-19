# Copyright 2024 Shunsuke Kitada and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script was generated from shunk031/cookiecutter-huggingface-datasets.
#

from __future__ import annotations

import ast
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, TypedDict, Union, cast

import datasets as ds
import pandas as pd
from datasets.utils.logging import get_logger
from PIL import Image
from PIL.Image import Image as PilImage

if TYPE_CHECKING:
    from ralfpt.saliency_detection import SaliencyTester
    from simple_lama_inpainting import SimpleLama

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


def ralf_style_example(
    example,
    inpainter: SimpleLama,
    saliency_testers: List[SaliencyTester],
    old_saliency_maps: Sequence[str] = (
        "basnet_saliency_map",
        "pfpn_saliency_map",
    ),
    new_saliency_maps: Sequence[str] = (
        "saliency_map",
        "saliency_map_sub",
    ),
):
    from ralfpt.inpainting import apply_inpainting
    from ralfpt.saliency_detection import apply_saliency_detection
    from ralfpt.transforms import has_valid_area, load_from_pku_ltrb
    from ralfpt.typehints import Element

    assert len(old_saliency_maps) == len(new_saliency_maps)
    assert len(new_saliency_maps) == len(saliency_testers)

    original_image = example["original_poster"]
    image_w, image_h = original_image.size

    poster_path = example["annotations"][0]["poster_path"]

    def get_pku_layout_elements(
        annotations, image_w: int, image_h: int
    ) -> List[Element]:
        ann, *_ = annotations
        total_elem = ann["total_elem"]

        elements = []
        for i in range(total_elem):
            cls_elem = annotations[i]["cls_elem"]
            box_elem = annotations[i]["box_elem"]

            label = cls_elem

            coordinates = load_from_pku_ltrb(
                box=box_elem, global_width=image_w, global_height=image_h
            )
            if has_valid_area(**coordinates):
                element: Element = {"label": label, "coordinates": coordinates}
                elements.append(element)

        return elements

    #
    # Remove the old saliency maps
    #
    for old_sal_map in old_saliency_maps:
        del example[old_sal_map]

    #
    # Get layout elements
    #
    try:
        elements = get_pku_layout_elements(
            annotations=example["annotations"], image_w=image_w, image_h=image_h
        )
    except AssertionError as e:
        logger.warning(f"[{poster_path}] Failed to get layout elements: {e}")

        # If the layout elements are not available,
        # return the example without inpainting and saliency maps
        for new_sal_map in new_saliency_maps:
            example[new_sal_map] = None

        return example

    #
    # Apply RALF-style inpainting
    #
    inpainted_image = apply_inpainting(
        image=original_image, elements=elements, inpainter=inpainter
    )
    example["inpainted_poster"] = inpainted_image

    #
    # Apply Ralf-style saliency detection
    #
    saliency_maps = apply_saliency_detection(
        image=inpainted_image,
        saliency_testers=saliency_testers,  # type: ignore
    )
    for new_sal_map, sal_map in zip(new_saliency_maps, saliency_maps):
        example[new_sal_map] = sal_map

    return example


@dataclass
class PosterLayoutConfig(ds.BuilderConfig):
    _saliency_maps: Optional[Sequence[str]] = None
    _saliency_testers: Optional[Sequence[str]] = None

    def get_default_salieny_maps(self) -> Sequence[str]:
        return ["basnet_saliency_map", "pfpn_saliency_map"]

    def get_salient_maps(self) -> Sequence[str]:
        if self.name == "default":
            return self.get_default_salieny_maps()
        elif self.name == "ralf":
            return ["saliency_map", "saliency_map_sub"]
        else:
            raise ValueError("Invalid config name")

    def get_saliency_testers(self) -> Optional[Sequence[str]]:
        if self.name == "ralf":
            return [
                "creative-graphic-design/ISNet-general-use",
                "creative-graphic-design/BASNet-SmartText",
            ]

    def __post_init__(self):
        super().__post_init__()
        self._saliency_maps = self.get_salient_maps()
        self._saliency_testers = self.get_saliency_testers()

    @property
    def saliency_maps(self) -> Sequence[str]:
        assert self._saliency_maps is not None
        return self._saliency_maps

    @property
    def saliency_testers(self) -> Sequence[str]:
        assert self._saliency_testers is not None
        return self._saliency_testers


class PosterLayoutDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = PosterLayoutConfig
    BUILDER_CONFIGS = [
        PosterLayoutConfig(name="default", version=VERSION),
        PosterLayoutConfig(name="ralf", version=VERSION),
    ]

    def _info(self) -> ds.DatasetInfo:
        base_features = {
            "original_poster": ds.Image(),
            "inpainted_poster": ds.Image(),
            "canvas": ds.Image(),
        }
        saliency_map_features = (
            {
                "basnet_saliency_map": ds.Image(),
                "pfpn_saliency_map": ds.Image(),
            }
            if self.config.name == "default"
            else {
                "saliency_map": ds.Image(),
                "saliency_map_sub": ds.Image(),
            }
        )
        annotation_features = {
            "annotations": ds.Sequence(
                {
                    "poster_path": ds.Value("string"),
                    "total_elem": ds.Value("int32"),
                    "cls_elem": ds.ClassLabel(
                        num_classes=4, names=["text", "logo", "underlay", "INVALID"]
                    ),
                    "box_elem": ds.Sequence(ds.Value("int32")),
                }
            ),
        }
        features = ds.Features(
            {**base_features, **saliency_map_features, **annotation_features}
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

        ann_df = ann_df.assign(
            # Convert string to list
            box_elem=ann_df["box_elem"].apply(ast.literal_eval),
            # Since PKU's label is 1-indexed, we need to convert it to 0-indexed
            cls_elem=ann_df["cls_elem"] - 1,
        )
        ann_df = ann_df.assign(
            cls_elem=ann_df["cls_elem"].replace(
                #
                # Convert class index to class name.
                #
                # The index = -1 produced by the conversion from 1-indexed to 0-indexed
                # is treated here as an INVALID class.
                #
                {-1: "INVALID", 0: "text", 1: "logo", 2: "underlay"}
            )
        )

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
            example = {
                "original_poster": load_image(original_poster_path),
                "inpainted_poster": load_image(inpainted_poster_path),
                "basnet_saliency_map": load_image(basnet_map_path),
                "pfpn_saliency_map": load_image(pfpn_map_path),
                "canvas": None,
                "annotations": annotations,
            }
            yield i, example

    def _generate_test_examples(self, poster: TestPoster, saliency_maps: SaliencyMaps):
        canvas_files = get_canvas_files(base_dir=poster["canvas"])

        basnet_map_files = get_basnet_map_files(base_dir=saliency_maps["basnet"])
        pfpn_map_files = get_pfpn_map_files(base_dir=saliency_maps["pfpn"])

        assert len(canvas_files) == len(basnet_map_files) == len(pfpn_map_files)
        it = zip(canvas_files, basnet_map_files, pfpn_map_files)

        for i, (canvas_path, basnet_map_path, pfpn_map_path) in enumerate(it):
            example = {
                "original_poster": None,
                "inpainted_poster": None,
                "basnet_saliency_map": load_image(basnet_map_path),
                "pfpn_saliency_map": load_image(pfpn_map_path),
                "canvas": load_image(canvas_path),
                "annotations": None,
            }
            yield i, example

    def _get_generator(
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

    def _generate_examples(
        self,
        poster: Union[TrainPoster, TestPoster],
        saliency_maps: SaliencyMaps,
        annotation: Optional[str] = None,
    ):
        config: PosterLayoutConfig = self.config  # type: ignore

        generator = self._get_generator(
            poster=poster,
            saliency_maps=saliency_maps,
            annotation=annotation,
        )

        def _generate_default(generator):
            for idx, example in generator:
                yield idx, example

        def _generate_ralf_style(generator):
            from ralfpt.saliency_detection import SaliencyTester
            from simple_lama_inpainting import SimpleLama

            inpainter = SimpleLama()
            saliency_testers = [
                SaliencyTester(model_name=model) for model in config.saliency_testers
            ]

            for idx, example in generator:
                old_saliency_maps = config.get_default_salieny_maps()
                new_saliency_maps = config.saliency_maps

                example = ralf_style_example(
                    example,
                    inpainter=inpainter,
                    old_saliency_maps=old_saliency_maps,
                    new_saliency_maps=new_saliency_maps,
                    saliency_testers=saliency_testers,
                )
                yield idx, example

        if config.name == "default":
            yield from _generate_default(generator)

        elif config.name == "ralf":
            yield from _generate_ralf_style(generator)

        else:
            raise ValueError(f"Invalid config name: {config.name}")
