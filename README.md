---
annotations_creators:
- expert-generated
language:
- zh
language_creators:
- found
license:
- cc-by-sa-4.0
multilinguality: []
pretty_name: PKU-PosterLayout
size_categories: []
source_datasets:
- extended|PosterErase
tags:
- layout-generation
- graphic design
task_categories:
- other
task_ids: []
---

# Dataset Card for PKU-PosterLayout

[![CI](https://github.com/shunk031/huggingface-datasets_PKU-PosterLayout/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/huggingface-datasets_PKU-PosterLayout/actions/workflows/ci.yaml)

## Table of Contents

- [Dataset Card Creation Guide](#dataset-card-creation-guide)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** http://59.108.48.34/tiki/PosterLayout/
- **Repository:** https://github.com/shunk031/huggingface-datasets_PKU-PosterLayout
- **Paper (Preprint):** https://arxiv.org/abs/2303.15937
- **Paper (CVPR2023):** https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html

### Dataset Summary

PKU-PosterLayout is a new dataset and benchmark for content-aware visual-textual presentation layout.

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages

The language data in PKU-PosterLayout is in Chinese ([BCP-47 zh](https://www.rfc-editor.org/info/bcp47)).

## Dataset Structure

### Data Instances

To use PKU-PosterLayout dataset, you need to download the poster image and saliency maps via [PKU Netdisk](https://disk.pku.edu.cn/link/999C6E97BB354DF8AD0F9E1F9003BE05) or [Google Drive](https://drive.google.com/drive/folders/1Gk202RVs9Qy2zbJUNeurC1CaQYNU-Vuv?usp=share_link).

```
/path/to/datasets
├── train
│   ├── inpainted_poster.zip
│   ├── original_poster.zip
│   ├── saliencymaps_basnet.zip
│   └── saliencymaps_pfpn.zip
└── test
    ├── image_canvas.zip
    ├── saliencymaps_basnet.zip
    └── saliencymaps_pfpn.zip
```

```python
import datasets as ds

dataset = ds.load_dataset(
    path="shunk031/PKU-PosterLayout",
    data_dir="/path/to/datasets/",
)
```

### Data Fields

[More Information Needed]

### Data Splits

[More Information Needed]

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

[More Information Needed]

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

[More Information Needed]

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

[More Information Needed]

### Citation Information

```bibtex
@inproceedings{hsu2023posterlayout,
  title={PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout},
  author={Hsu, Hsiao Yuan and He, Xiangteng and Peng, Yuxin and Kong, Hao and Zhang, Qing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6018--6026},
  year={2023}
}
```

### Contributions

Thanks to [@PKU-ICST-MIPL](https://github.com/PKU-ICST-MIPL) for creating this dataset.
