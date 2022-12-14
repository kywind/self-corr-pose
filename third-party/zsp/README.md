# Zero-Shot Category-Level Object Pose Estimation

This repository contains Pytorch code for the paper **Zero-Shot Category-Level Object Pose Estimation (Goodwin et al., 2020)** [[arxiv]](https://arxiv.org/abs/2204.03635).

![alt text](main.jpg?raw=true "Zero-Shot Category-Level Object Pose Estimation")

## Installation
* Make environment:
`conda env create -f environment.yml`

* Install Pillow < 7.0 with `pip` to overcome a `torchvision` bug:
`pip install 'pillow<7'`

* Install Pytorch3D from Github:
`pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"`

Install the `zsp` python package implemented in this repo with `pip install -e .`

## Dataset
This work uses the Common Objects in 3D (CO3D) dataset. The repo for this dataset, with download instructions, is [here](https://github.com/facebookresearch/co3d). 

This dataset contains 18,619 multi-frame sequences capturing different instances of 50 object categories. For full dataset is around 1.4TB. For evaluation in this work, we manually annotated 10 sequences from each of 20 categories with ground-truth poses (these annotations are found under `data/class_labels`). The relevant subset of the dataset is thus smaller at around ~15GB. If you are struggling to download the entire CO3D dataset, please contact me and I will try to share this subset with you.

## Pre-trained models
This code uses DINO ViTs for feature extraction. Links to pre-trained weights can be found in [this](https://github.com/facebookresearch/dino/blob/main/hubconf.py) file. However, to just download the main model considered in this work:
```
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth
```
The directory to which you save this model can be passed as an argument to the main script.

## Running the code
```
cd zsp
python method/evaluate_ref_to_target_pose.py \
    --co3d_root /path/to/co3d/dataset \
    --hub_dir /path/to/saved/dino/weights/ \
    --kmeans 
```
By default, this will loop over the 20 categories in the labelled subset developed in this work, and draw 100 reference-target pairings from the 10 labelled sequences in each of these categories. To vary the number of target frames used (default = 5), change the `--n_target` argument.

To plot results (correspondences, the closest matching frame, and renders of the aligned point clouds), pass `--plot_results`.

## Citation

If you use this code in your research, please consider citing our paper:
```
@article{goodwin2022,
    author  = {Walter Goodwin and Sagar Vaze and Ioannis Havoutis and Ingmar Posner},
    title   = {Zero-Shot Category-Level Object Pose Estimation},
    journal = {arXiv preprint},
    year    = {2022},
  }
```

