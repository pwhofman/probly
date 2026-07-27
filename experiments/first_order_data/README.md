# First-order Data Experiments

Run commands in this directory:

```bash
cd experiments/first_order_data
uv sync
```

By default, downloaded datasets are stored in `~/datasets/dcic`, shared with the `probly_benchmark` training pipeline (see below). Pass `--dest` to [install_dcic_datasets.py](install_dcic_datasets.py) to use a different location.

## Image dataset overview (DCIC Datasets)

| Dataset       |  size | classes | avg. entropy | avg. non-zero target classes | type                                                                | input size |
| ------------- | ----: | ------: | -----------: | ---------------------------: | ------------------------------------------------------------------- | :--------: |
| Benthic       |  4867 |       8 |        0.340 |                         1.64 | images from the seafloor and consists of underwater flora and fauna |  112x112   |
| CIFAR-10H     | 10000 |      10 |        0.154 |                         1.95 | reannotated variant of the original CIFAR-10 test set               |   32x32    |
| MiceBone      |  7240 |       3 |        0.319 |                         1.56 | Second-Harmonic-Generation images of collagen fibers                |  224x224   |
| Pig           | 10237 |       4 |        0.735 |                         2.57 | tail images form european farms                                     |   96x96    |
| Plankton      | 12280 |      10 |        0.163 |                         1.50 | underwater plankton images                                          |   96x96    |
| QualityMRI    |   310 |       2 |        0.556 |                         2.00 | MRI images                                                          |  224x224   |
| Synthetic     | 15000 |       6 |        0.584 |                         2.66 | images that contain 1 colored circle on a black background          |  224x224   |
| TreeVersity#1 |  9489 |       6 |        0.266 |                         1.68 | plant images, single label per image                                |  224x224   |
| TreeVersity#6 |  9826 |       6 |        0.742 |                         3.11 | plant images, possibly multiple labels per image                    |  224x224   |
| Turkey        |  8040 |       3 |        0.196 |                         1.44 | images of turkeys and their injuries                                |  192x192   |

`avg. non-zero target classes` is the mean number of classes with non-zero target probability per image.

Image datasets should be stored under `~/datasets/dcic`. Used datasets can be downloaded here: https://zenodo.org/records/8115942 or directly with the [install_dcic_datasets.py](install_dcic_datasets.py) helper.

<details>
<summary><strong>Detailed dataset descriptions</strong></summary>

The Data-Centric Image Classification (DCIC) benchmark [[Schmarje et al., 2022]][schmarje2022benchmark] is a set of ten datasets from diverse domains, including natural, biological, medical and synthetic images. The benchmark uses ambiguous and noisy image labels through multiple annotations per image. Each annotation has an image path, an assigned class label, and metadata such as the creation date. In the released benchmark version, all datasets follow the same substructure and are provided with predefined five-fold splits.

<!-- Large parts of the information are from the benchmark paper: https://arxiv.org/pdf/2207.06214 (section 2, Datasets and detailed dataset descriptions found in Supplementary section 6 + 7) -->

#### Benthic

Benthic consists of seafloor imagery showing underwater flora and fauna. The task is to classify the main seafloor object. It has eight classes in total, including "coral", "crustacean", "cucumber", "encrusting", "other_fauna", "sponge", "star", and "worm".
The original data and annotations were introduced by [[Langenkämper et al., 2020]][Langenkamper2020GearStudy] [[Schoening et al., 2020]][schoening2020Megafauna]. The benchmark version only keeps objects with at least three annotations and crops the original images around the annotated object.

#### CIFAR-10H

CIFAR-10H [[Peterson et al., 2019]][peterson2019cifar10h] is a reannotated variant of the original CIFAR-10 [[Krizhevsky and Hinton, 2009]][krizhevsky2009learning] test set. The goal is a standard CIFAR-10 object classification. The benchmark version didn't modify the data.

#### MiceBone

MiceBone is made up of microscope images from mouse tissue, where the imaging technique is designed to reveal collagen fibers. The task is to classify the collagen-fiber orientation structure. The dataset has 3 classes: similar/aligned collagen fiber orientations ("g"), dissimilar collagen fiber orientations ("ug"), and not relevant regions due to noise or background ("nr"). The original labels were introduced by [[Schmarje et al., 2019]][schmarje2019]. The DCIC benchmark version expanded the annotation set using additional human annotators and performed additional preprocessing as described in [[Schmarje et al., 2022 (ECCV)]][tiho_mods_00007665].

<!-- 1 domain expert, others paid workers -->
<!-- Preprocessing: "The raw data are 3D scans from collagen fibers in mice bones. The three proposed classes are similar and dissimilar collagen fiber orientations and not relevant regions due to noise or background. We used the given segmentations to cut image regions from the original 2D image slices which mainly consist of one class.", https://zenodo.org/records/7152298 -->

#### Pig

Pig consists of cropped pig tail images from European farms, with varying injury levels. The task is to classify the tail injury severity. The dataset has 4 classes: "1_intact" (no injury), "2_short" (shortened but healed), "3_fresh" (fresh wound), "4_notVisible" (tail not visible).
The raw cropped images were sourced from the University of Helsinki [[Schmarje et al., 2022]][schmarje2022benchmark]. The DCIC benchmark authors sourced all annotations through hired workers with relevant domain knowledge.

<!-- 5 domain experts + 1 hired worker -->

#### Plankton

Plankton is a collection of underwater plankton images. The task is to classify the plankton images into organism or morphotype categories. The dataset has 10 classes: "bubbles", "collodaria_black", "collodaria_globule", "cop", "det", "no_fit", "phyto_puff", "phyto_tuft", "pro_rhizaria_phaeodaria" and "shrimp". The data was collected in [[Schmarje et al., 2021]][schmarje2021foc] and later adjusted in [[Schmarje et al., 2022 (DC3)]][schmarje2022dc3], with annotations stemming from citizen scientists. The DCIC benchmark performed additional preprocessing as described in [[Schmarje et al., 2022 (ECCV)]][tiho_mods_00007665] and didn't enforce any class balance in the folds for this dataset.

<!-- Original dataset was collected in first cited paper, then adjusted in the 2022 paper, which is also the version used here. Class balance not enforced for this dataset in individual folds by authors. -->
<!-- Preprocessing: "we preprocessed the data by recentering the images and removing artifacts like scale bars.", https://zenodo.org/records/7152298 -->

#### QualityMRI

QualityMRI consists of human MRI images with varying image quality. In the DCIC benchmark, the task is formulated as a binary classification between "0" (lower image quality) and "1" (higher image quality). The dataset and annotations were sourced from two prior studies: Obuchowicz et al. [[Obuchowicz et al., 2020]][obuchowicz2020qualityMRI] collected 70 MRI images and Stepien et al. [[Stępień et al., 2021]][stepien2021cnnQuality], who collected an additional 240 MRI images. In both studies, image quality assessment was done by radiologists on a scale from 1 to 5.
The DCIC benchmark binarized the joint dataset of 310 annotated MRI images as follows: a score of 1 was mapped to four 0s, a score of 2 to three 0s and one 1, and so forth, until a score of 5, which was converted to four 1s.

#### Synthetic

Synthetic consists of synthetically generated images with a colorful circle or ellipse on a black background. The task is to classify the color-shape combination of a single object. The classes include: "bc" (blue circle), "be" (blue ellipse), "gc" (green circle), "ge" (green ellipse), "rc" (red circle) and "re" (red ellipse). This dataset was fully generated solely for the DCIC benchmark.

#### TreeVersity

TreeVersity is a dataset of plant images with multiple possible tags per image. The task is to classify each image into one or more of the following labels: "bark", "bud", "flower", "fruit", "leaf" and "whole_plant". The dataset and annotations were crowdsourced from the Arnold Arboretum of Harvard University [[Arnold Arboretum]][arnoldArboretumDataResources]. Originally the dataset had 22 possible tags. The DCIC benchmark simplified the dataset to only have six tags. The DCIC benchmark created two versions of the dataset: Treeversity#1, where images have exactly one user-assigned label per image and Treeversity#6 where images can have multiple labels (up to six) per image.

#### Turkey

Turkey consists of images of turkeys and their injuries. The task is to classify the injury type from the images. The dataset has three classes: "head_injury" (injuries from neck upwards), "plumage_injury" (injuries elsewhere on body) and "not_injured" (no visible injury). The original data was introduced by Volkmann et al. [[Volkmann et al., 2021]][volkmann2021turkeys], who collected over 19,500 annotated images. A follow-up study by Volkmann et al. [[Volkmann et al., 2022]][volkmann2022keypoint] added body-keypoint annotations for 244 images containing 7,660 turkeys, with seven keypoints per bird. For the DCIC benchmark, the original turkey-injury data label distributions were further preprocessed [[Schmarje et al., 2022 (ECCV)]][tiho_mods_00007665] and the annotation set was expanded by a factor of five using hired workers.

#### References

- **[schmarje2022benchmark]** Schmarje, L., Grossmann, V., Zelenka, C., Dippel, S., Kiko, R., Oszust, M., Pastell, M., Stracke, J., Valros, A., Volkmann, N., Koch, R. (2022). *Is one annotation enough? A data-centric image classification benchmark for noisy and ambiguous label estimation.* NeurIPS 2022 Datasets and Benchmarks Track. <https://arxiv.org/abs/2207.06214>
- **[schoening2020Megafauna]** Schoening, T., Purser, A., Langenkämper, D., et al. (2020). *Megafauna community assessment of polymetallic-nodule fields with cameras: platform and methodology comparison.* Biogeosciences 17(12), 3115–3133. <https://doi.org/10.5194/bg-17-3115-2020>
- **[Langenkamper2020GearStudy]** Langenkämper, D., van Kevelaer, R., Purser, A., Nattkemper, T. W. (2020). *Gear-Induced Concept Drift in Marine Images and Its Effect on Deep Learning Classification.* Frontiers in Marine Science 7. <https://doi.org/10.3389/fmars.2020.00506>
- **[peterson2019cifar10h]** Peterson, J., Battleday, R., Griffiths, T., Russakovsky, O. (2019). *Human uncertainty makes classification more robust.* ICCV 2019, 9616–9625. <https://doi.org/10.1109/ICCV.2019.00971>
- **[krizhevsky2009learning]** Krizhevsky, A., Hinton, G. (2009). *Learning multiple layers of features from tiny images.* Technical report, University of Toronto. <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>
- **[schmarje2019]** Schmarje, L., Zelenka, C., Geisen, U., Glüer, C.-C., Koch, R. (2019). *2D and 3D Segmentation of uncertain local collagen fiber orientations in SHG microscopy.* DAGM GCPR 2019, 374–386. <https://doi.org/10.1007/978-3-030-33676-9_26>
- **[schmarje2021foc]** Schmarje, L., Brünger, J., Santarossa, M., Schröder, S.-M., Kiko, R., Koch, R. (2021). *Fuzzy Overclustering: Semi-Supervised Classification of Fuzzy Labels with Overclustering and Inverse Cross-Entropy.* Sensors 21(19), 6661. <https://doi.org/10.3390/s21196661>
- **[schmarje2022dc3]** Schmarje, L., Santarossa, M., Schröder, S.-M., Zelenka, C., Kiko, R., Stracke, J., Volkmann, N., Koch, R. (2022). *A data-centric approach for improving ambiguous labels with combined semi-supervised classification and clustering.* ECCV 2022.
- **[obuchowicz2020qualityMRI]** Obuchowicz, R., Oszust, M., Piorkowski, A. (2020). *Interobserver variability in quality assessment of magnetic resonance images.* BMC Medical Imaging 20(1), 109. <https://doi.org/10.1186/s12880-020-00505-z>
- **[stepien2021cnnQuality]** Stępień, I., Obuchowicz, R., Piórkowski, A., Oszust, M. (2021). *Fusion of Deep Convolutional Neural Networks for No-Reference Magnetic Resonance Image Quality Assessment.* Sensors 21(4). <https://doi.org/10.3390/s21041043>
- **[arnoldArboretumDataResources]** Arnold Arboretum of Harvard University. *Data Resources.* Accessed 2026-03-21. <https://arboretum.harvard.edu/research/data-resources/>
- **[volkmann2021turkeys]** Volkmann, N., Brünger, J., Stracke, J., Zelenka, C., Koch, R., Kemper, N., Spindler, B. (2021). *Learn to train: Improving training data for a neural network to detect pecking injuries in turkeys.* Animals 11, 1–13. <https://doi.org/10.3390/ani11092655>
- **[volkmann2022keypoint]** Volkmann, N., Zelenka, C., Devaraju, A. M., Brünger, J., Stracke, J., Spindler, B., Kemper, N., Koch, R. (2022). *Keypoint Detection for Injury Identification during Turkey Husbandry Using Neural Networks.* Sensors 22(14), 5188. <https://doi.org/10.3390/s22145188>
- **[tiho_mods_00007665]** Schmarje, L., Santarossa, M., Schröder, S.-M., Zelenka, C., Kiko, R., Stracke, J., Volkmann, N., Koch, R. (2022). *A data-centric approach for improving ambiguous labels with combined semi-supervised classification and clustering.* ECCV 2022, Part VIII, 363–380, Springer. <https://doi.org/10.1007/978-3-031-20074-8_21>

[schmarje2022benchmark]: https://arxiv.org/abs/2207.06214
[schoening2020Megafauna]: https://doi.org/10.5194/bg-17-3115-2020
[Langenkamper2020GearStudy]: https://doi.org/10.3389/fmars.2020.00506
[peterson2019cifar10h]: https://doi.org/10.1109/ICCV.2019.00971
[krizhevsky2009learning]: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
[schmarje2019]: https://doi.org/10.1007/978-3-030-33676-9_26
[schmarje2021foc]: https://doi.org/10.3390/s21196661
[schmarje2022dc3]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880353.pdf
[obuchowicz2020qualityMRI]: https://doi.org/10.1186/s12880-020-00505-z
[stepien2021cnnQuality]: https://doi.org/10.3390/s21041043
[arnoldArboretumDataResources]: https://arboretum.harvard.edu/research/data-resources/
[volkmann2021turkeys]: https://doi.org/10.3390/ani11092655
[volkmann2022keypoint]: https://doi.org/10.3390/s22145188
[tiho_mods_00007665]: https://doi.org/10.1007/978-3-031-20074-8_21

</details>

## Credal-set predictor training & evaluation

Training uses the main `probly_benchmark` package, not this folder — run these commands from the **repo root** (`uv run ...`), not from `experiments/first_order_data`.

- Training entrypoint: [`src/probly_benchmark/train.py`](../../src/probly_benchmark/train.py), driven by Hydra `method`/`recipe` configs.
- Recipe configs exist for all 10 DCIC datasets: `src/probly_benchmark/configs/recipe/resnet50_<dataset>.yaml` (e.g. `resnet50_micebone`).
- For the paper, 3 credal-set methods and 3 datasets were selected: `credal_ensembling` (CreEns), `credal_relative_likelihood` (CreRL), `credal_wrapper` (CreWra), on **micebone**, **qualitymri**, and **synthetic**.

```bash
# from the repo root
uv run python src/probly_benchmark/train.py method=credal_ensembling recipe=resnet50_micebone
```

Training logs to Weights & Biases (`wandb.entity`/`wandb.project` in `configs/train.yaml`); each run's `run_id` is needed for evaluation below.

### Evaluation

[`paper_comparisons.py`](paper_comparisons.py) evaluates trained runs and produces LaTeX tables. It lives in this folder but depends on `probly_benchmark`, so it must also be run from the **repo root**. Fill in the `WANDB_RUN_IDS` dict at the top of the file (`(method, dataset) -> wandb run_id`) with the runs you want to compare, then:

```bash
# from the repo root
uv run python experiments/first_order_data/paper_comparisons.py
```

Outputs (under `experiments/first_order_data/first_order_results_probly/`, gitignored):

- `results.json` + `results_arrays.npz`: raw scalar and per-instance metrics
- `plots/`: per-(dataset, method) L1-calibration histograms and a CP set-size-vs-coverage-violation scatter grid
- `tables.tex` (also printed to stdout): three tables —
  1. credal-set coverage/efficiency (CH Cov / Int. Cov. / Efficiency)
  2. calibration (L1 / ECE)
  3. conformal prediction with LAC/APS/RAPS/SAPS (Cond. Sat. / Set Size / Marg. Cov.)
