<h2 align="center">Identifying Optimal nnU-Net Configuration for Cerebral Microbleed Segmentation</h2>

<p align="center">
  <b>Junmo Kwon<sup>1</sup>, Sang Won Seo<sup>2</sup>, Hwan-ho Cho<sup>3</sup>, Hyunjin Park<sup>1</sup></b>
</p>

<p align="center">
  <sup>1</sup>Department of Electrical and Computer Engineering, Sungkyunkwan University, Suwon 16419, South Korea<br>
  <sup>2</sup>Department of Neurology, Samsung Medical Center, Sungkyunkwan University School of Medicine, Seoul 06351, South Korea<br>
  <sup>3</sup>Department of Electronics Engineering, Incheon National University, Incheon 22012, South Korea<br>
</p>

# Setting Up nnU-Net
Follow the [**installation guide**](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) to install **nnUNet-v2**.

# Adding Custom CMB Preprocessing
Download the two files to setup our CMB preprocessing framework:
1. Download [``local_minima_preprocessor.py``](./local_minima_preprocessor.py) to ``nnunetv2/preprocessing/preprocessors/local_minima_preprocessor.py``.
2. Download [``local_minima_planners.py``](./local_minima_planners.py) to ``nnunetv2/experiment_planning/experiment_planners/residual_unets/local_minima_planners.py``.

# Running Custom nnUNet-v2
## Building up training dataset
The CMB training dataset requires co-registered T1- and T2*-weighted MRI scans for every subject. Your `dataset.json` should follow this format:

```
{
    "channel_names": {
        "0": "T2S",
        "1": "T1"
    },
    "file_ending": ".nii.gz",
    "labels": {
        "CMB": 1,
        "background": 0
    },
    "numTraining": 0,
    "overwrite_image_reader_writer": "SimpleITKIO"
}
```

where `numTraining` should match the size of your training dataset.

Please ensure the channel name of T2* MRI scan is set to `T2S`.

This is necessary for [``local_minima_preprocessor.py``](./local_minima_preprocessor.py) to process minimum filtering and local minima extraction.

## Planning and Preprocessing
[``local_minima_planners.py``](./local_minima_planners.py) offers three input patch size:
1. `LocalMinimaResEncL24` for 24×448×384
2. `LocalMinimaResEncL16` for 16×224×192
3. `LocalMinimaResEncL12` for 12×112×96

Always start by running the default experiment planning on your dataset. Then, adjust the input patch size to your desired size.

```
import os

working_dir = 'path/to/working/directory'
os.environ['nnUNet_raw'] = os.path.join(working_dir, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(working_dir, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(working_dir, 'nnUNet_results')

dataset_id = 1  # your dataset id
plan = 'LocalMinimaResEncL24'
preprocessor = 'LocalMinimaPreprocessor'
!nnUNetv2_plan_and_preprocess -c 3d_fullres -d "{dataset_id}" -pl "{plan}" -preprocessor_name "{preprocessor}" --verbose
```
## Training nnUNet-v2
```
dataset_id = 1  # your dataset id
plan = 'LocalMinimaResEncL24'  # your plan name
for fold in (0, 1, 2, 3, 4):
    !CUDA_VISIBLE_DEVICES=0 nnUNetv2_train "{dataset_id}" 3d_fullres "{fold}" -p "{plan}" --c
```
## Inference with nnUNet-v2
```
dataset_id = 1  # your dataset id
plan = 'LocalMinimaResEncL24'  # your plan name
input_path = 'path/to/test/set'
output_path = 'path/to/output/path'
for fold in (0, 1, 2, 3, 4):
    !CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -d "{dataset_id}" -c 3d_fullres -f "{fold}" -i "{input_path}" -o "{output_path}" -p "{plan}"
```

## Extending to Multi-class Segmentation
The current [``local_minima_preprocessor.py``](./local_minima_preprocessor.py) only supports binary CMB segmentation and should not be used for multi-class segmentation.

Attempting to use it for multi-class segmentation will trigger the following assertion error:
```
assert len(classes_or_regions) == 1, f"LocalMinimaPreprocessor currently supports only binary segmentation whereas the given data has {classes_or_regions} classes"
```
The tricky part is that `classes_or_regions` could be either class or region, where region denotes [region-based training](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md).

If you are absolutely certain that `classes_or_regions` always indicates classes, then you may try to adapt the local minima preprocessing as follows.
1. Determine the class index that represents CMB (by reading the `dataset_json`)
2. Apply the local minima preprocessing to the given CMB index
3. Apply the conventional foreground sampling to all remaining class indices.

# Acknowledgement
Part of the codes are referred from the following open-source projects:

* https://github.com/MIC-DKFZ/nnUNet
