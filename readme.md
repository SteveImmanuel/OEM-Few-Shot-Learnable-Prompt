# Learnable Prompt for Few-Shot Semantic Segmentation
This repo contains the code for "OpenEarthMap Land Cover Mapping Few-Shot Challenge" co-organized with the L3D-IVU CVPR 2024 Workshop.

## Setup
Clone the repository using:
```bash
git clone --recursive git://github.com/foo/bar.git
```

This code is developed with Python 3.9.

### PIP
Install the required packages by running:
```bash
pip install -r requirements.txt
```

### Conda (Only for Linux)
Create a new conda environment and install the required packages by running:
```bash
conda env create -f env.yml
```

## Dataset
Download the dataset from https://zenodo.org/records/10828417.

### Training Set (Base Classes)
Extract the `trainset.zip` and configure the directory as follows:
```
trainset
├── images
│   ├── aachen_20.tif
│   ├── aachen_3.tif
│   ...
└── labels
    ├── aachen_20.tif
    ├── aachen_3.tif
    ...
```

Convert the labels into colored semantic segmentation masks using the `prepare_dataset.py` script:
```bash
python prepare_dataset.py --dataset-dir <path/to/trainset> --convert-color
```
Verify that there is a new directory called `labels_color` containing the semantic segmentation masks.

### Validation and Test Set (Novel Classes)
Extract the `valset.zip` and `testset.zip` and configure the directories as follows:
```
testset
├── images
│   ├── aachen_17.tif
│   ├── aachen_38.tif
│   ...
└── labels
    ├── aachen_17.tif
    ├── aachen_38.tif
    ...
```

Run the `prepare_dataset.py` script to configure the directories as follows:
```bash
python prepare_dataset.py --dataset-dir <path/to/testset>
```

Make sure that the final directories looks like this:
```
testset
├── 10
│   ├── images
│   │   ├── ica_14.tif
│   │   ...
│   └── labels
│       ├── ica_14.tif
│       ...
├── 11
│   ├── images
│   │   ├── koeln_58.tif
│   │   ...
│   └── labels
│       ├── koeln_58.tif
│       ...
├── 8
│   ├── images
│   │   ├── austin_34.tif
│   │   ...
│   └── labels
│       ├── austin_34.tif
│       ...
├── 9
│   ├── images
│   │   ├── aachen_17.tif
│   │   ...
│   └── labels
│       ├── aachen_17.tif
│       ...
└── images
    ├── aachen_38.tif
    ├── aachen_44.tif
    ...
```
The directory `8` to `11` contains the `support set` each for the respective novel classes. The `images` directory contains the `query set`.

## Training
Training is separated for base classes and novel classes.
Make sure to change the `train_dataset_dir` and `val_dataset_dir` in the config files.

**Note**: Validation set requires full labels for all classes, meanwhile the `support set` only contains the label novel specific novel class. We use the same training set for validation.

### Base Classes
```bash
python train.py --config configs/base.json
```

### Novel Classes
Each novel class is trained independently, creating a prompt which amounts to ~5 MB checkpoint. 

Novel classes training consists of two phases. In the first phase, we train only with image regions that contains the novel class. In the second phase, we train with all image regions. Change the `model_path` in the config files to the checkpoint from the base classes training and `train_dataset_dir` and `val_dataset_dir` to the corresponding support set.

Phase 1
```bash
python train_adapter.py --config configs/adapter_<i>.json
```
Phase 2
```bash
python train_adapter.py --config configs/adapter_<i>.json --adapter-path <path/to/checkpoint/from/phase1> --phase-2 --lr 1e-5
```

## Inference

### Mappings for Prompt
We provide the mappings in the `mappings` directory. In order to generate these files, use the `notebooks/create_mapping.ipynb` and `notebooks/novel_class_filtering.ipynb` notebook.

### Base Classes
```bash
python inference.py --model-path <path/to/base/checkpoint> \
--prompt-img-dir <path/to/trainset/images> \
--prompt-label-dir <path/to/trainset/labels_color> \
--dataset-dir <path/to/queryset> \
--mapping mappings/test/vit.json \
--outdir <path/where/to/output>
```

### Novel Classes
Novel classes inference must be run for each novel class independently.
```bash
python inference_adapter.py --base-model-path <path/to/base/checkpoint> \
--adapter-model-path <path/to/specific/novel/idx/checkpoint> \
--class-idx <novel_class_idx> \
--outdir <path/where/to/output>
```

### Combining Base and Novel Classes
The results for base and novel classes can be combined using `combine.py` script. The script works by overlaying the novel classes predictions on top of the base classes predictions, so they need to be run for each novel class, one after the other.
```bash
python combine.py --class-idx <novel_class_idx> \
--outdir <path/where/to/output> \
--source-folder <path/to/base/predictions> \
--target-folder <path/to/novel/predictions> \
```
In subsequent runs, the `source-folder` should be the output directory of the previous run.

## Reproduce Submission
Setup the dataset directory and train all base and novel classes following the previous sections.

Due to stochasticity, we also provide all of our checkpoints for the base and novel classes.
Download the checkpoints `base.pt`, `8_test.pt`, `9_test.pt`, `10_test.pt`, `11_test.pt` from [here](https://drive.google.com/drive/folders/1). Place all of them in one directory.

Run the following command to generate the submission:
```bash
python submission.py \
--ckpt-path <path/to/checkpoints> \
--dataset-dir <path/to/query/set> \
--prompt-img-dir <path/to/trainset/images> \
--prompt-label-dir <path/to/trainset/labels_color> \
```
**Note**: The inference script requires ~16GB GPU memory.

This will generate the predictions for the base classes, each independent novel class, and the result of overlaying the novel classes to the base classes inside `out` directory. Finally, the file `out/submission.zip` contains the final prediction.

## Credits
This codebase is developed upon [SegGPT](https://github.com/baaivision/Painter).