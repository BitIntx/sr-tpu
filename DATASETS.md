# Super-Resolution Dataset Notes

## Local Dataset Root

```text
~/datasets/sr/
```

## Collected Locally

### DIV2K

Official page: https://data.vision.ee.ethz.ch/cvl/DIV2K/

Purpose: first classical single-image super-resolution baseline.

License note from the official page: academic research use only; images were
collected from the internet and copyright belongs to original owners.

Files collected:

```text
~/datasets/sr/archives/DIV2K_train_HR.zip
~/datasets/sr/archives/DIV2K_valid_HR.zip
~/datasets/sr/archives/DIV2K_train_LR_bicubic_X4.zip
~/datasets/sr/archives/DIV2K_valid_LR_bicubic_X4.zip
```

Expected extracted folders:

```text
~/datasets/sr/DIV2K/DIV2K_train_HR/
~/datasets/sr/DIV2K/DIV2K_valid_HR/
~/datasets/sr/DIV2K/DIV2K_train_LR_bicubic/X4/
~/datasets/sr/DIV2K/DIV2K_valid_LR_bicubic/X4/
```

Counts:

```text
DIV2K_train_HR: 800 PNG images
DIV2K_valid_HR: 100 PNG images
DIV2K_train_LR_bicubic/X4: 800 PNG images
DIV2K_valid_LR_bicubic/X4: 100 PNG images
```

Project symlinks:

```text
~/dev/sr-tpu/data/DIV2K_train_HR
~/dev/sr-tpu/data/DIV2K_valid_HR
~/dev/sr-tpu/data/DIV2K_train_LR_bicubic
~/dev/sr-tpu/data/DIV2K_valid_LR_bicubic
```

Archive checksums:

```text
~/datasets/sr/archives/SHA256SUMS
```

### OST / OutdoorScene

Project page: http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/

Purpose: larger natural outdoor-scene HR image pool for synthetic degradation
training. This is useful after DIV2K when the model needs more varied textures
than the 800-image classical training set.

Files collected:

```text
~/datasets/sr/OST/datasets/OutdoorSceneTrain_v2/*.zip
~/datasets/sr/OST/datasets/OutdoorSceneTest300/*.zip
~/datasets/sr/OST/datasets/OutdoorSeg/*.zip
```

Extracted folders:

```text
~/datasets/sr/OST/extracted/OutdoorSceneTrain_v2/
~/datasets/sr/OST/extracted/OutdoorSceneTest300/OutdoorSceneTest300/
~/datasets/sr/OST/extracted/OutdoorSeg/images/
```

Counts:

```text
OutdoorSceneTrain_v2/animal: 2187 PNG images
OutdoorSceneTrain_v2/building: 2285 PNG images
OutdoorSceneTrain_v2/grass: 1140 PNG images
OutdoorSceneTrain_v2/mountain: 1092 PNG images
OutdoorSceneTrain_v2/plant: 1036 PNG images
OutdoorSceneTrain_v2/sky: 1727 PNG images
OutdoorSceneTrain_v2/water: 857 PNG images
OutdoorSceneTest300: 300 PNG images
OutdoorSeg/images: 9900 JPG images
```

Note: `OutdoorSceneTest300_anno` and `OutdoorSeg/annotations` were also
extracted, but they are annotation masks, not HR photos for SR training.

Project symlinks:

```text
~/dev/sr-tpu/data/OST_train
~/dev/sr-tpu/data/OST_test300
~/dev/sr-tpu/data/OST_seg_images
```

Archive checksums:

```text
~/datasets/sr/OST/SHA256SUMS
```

## Prepared Training Dataset

### sr_x4_v1

Local path:

```text
~/datasets/sr/prepared/sr_x4_v1/
```

Purpose: first x4 SR training layout for the JAX/TPU scaffold. Training LR is
generated on the fly from HR random crops; validation LR/HR pairs are fixed and
stored under `val_x4_bicubic/`.

Filtering:

```text
min_side >= 256 pixels
scale = 4
validation HR images are modcropped to multiples of 4
```

Folders:

```text
train_clean_hr: 9784 images
train_balanced_hr: 12184 symlink entries
train_mixed_hr: 18919 symlink entries
val_hr: 398 images
val_x4_bicubic/hr: 398 PNG images
val_x4_bicubic/lr: 398 PNG images
val_x4_bicubic/lr_up: 398 PNG images
```

Default training folder:

```text
~/datasets/sr/prepared/sr_x4_v1/train_balanced_hr
```

Notes:

```text
train_balanced_hr = DIV2K train repeated 4x + OST train PNG
train_clean_hr = DIV2K train + OST train PNG, no repeats
train_mixed_hr = train_clean_hr + OST segmentation JPEG images
```

Rebuild command:

```bash
cd ~/dev/sr-tpu
python prepare_dataset.py --overwrite
```

## Good Next Datasets

### Flickr2K / DF2K

Purpose: larger classical SR training set. DF2K usually means DIV2K + Flickr2K.

Status: useful, but the official direct link was not reachable from this VM
during setup, a documented Hugging Face mirror returned 401 without credentials,
and there is no Kaggle token configured here. Common research codebases refer to
Flickr2K as 2,650 2K-resolution training images.

### RealSR

Official repository: https://github.com/csjcai/RealSR

Purpose: real-world paired LR/HR super-resolution from camera zoom captures.

Status: download links are Google Drive/Baidu from the repository. This is a
better second-stage dataset if the target is real photos rather than bicubic SR.

### Manga109

Official site: https://www.manga109.org/en/

Purpose: manga/comic line-art super-resolution evaluation and domain tuning.

Status: requires application through the official site. Do not scrape or mirror
without following the dataset terms.

## Practical Training Order

1. Start with DIV2K HR and generate LR on the fly.
2. Add official DIV2K LR x4 for validation against the standard degradation.
3. Add OST HR photos for more varied natural textures.
4. Add Flickr2K/DF2K after confirming source and terms.
5. Add a domain dataset for the intended use: photos, anime, manga, game textures,
   scanned documents, etc.
