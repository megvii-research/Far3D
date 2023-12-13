# Get Started
Note our Far3D inherit from the repo of [StreamPETR](https://github.com/exiawsh/StreamPETR/), thus many parts can be used in similar style. 

## 1. Setup
Please following [Setup instructions](https://github.com/exiawsh/StreamPETR/blob/main/docs/setup.md) to build environment and compile mmdet3d. We also provide detailed conda environment file [here](../py38.yaml).

**Instruction Steps**
1. Create a conda virtual environment and activate it.
2. Install PyTorch and torchvision following the official instructions.
3. Install flash-attn (optional).
4. Clone Far3D.
5. Install mmdet3d.

## 2. Data preparation
We use Argoverse 2 dataset and NuScenes dataset for experiments.
- NuScenes dataset preparation can refer to [Preparation](https://github.com/exiawsh/StreamPETR/blob/main/docs/data_preparation.md).
- Similarly, after downloading [Argoverse 2 sensor dataset](https://www.argoverse.org/av2.html#download-link), it can be processed following above pipelines and [create_av2_infos.py](../tools/create_infos_av2/create_av2_infos.py).
```angular2html
# first modify args such as split, dataset_dir.
python tools/create_infos_av2/gather_argo2_anno_feather.py
python tools/create_infos_av2/create_av2_infos.py
```

**Notes**: 
- Due to the huge strorage of Argoverse 2 dataset, we read data from s3 path. If any need to load from local disk or other paths, please modify [AV2LoadMultiViewImageFromFiles](../projects/mmdet3d_plugin/datasets/pipelines/custom_pipeline.py) for your convenience.
- For Argoverse 2, its 3D labels are in ego coordinates. The ego-motion transformation refer to [Argoverse2DatasetT](projects/mmdet3d_plugin/datasets/argoverse2_dataset_t.py).

## 3. Training & Inference
Train the model
```angular2html
tools/dist_train.sh projects/configs/far3d.py 8 --work-dir work_dirs/far3d/
```
Evaluation
```angular2html
tools/dist_test.sh projects/configs/far3d.py work_dirs/far3d/iter_82548.pth 8 --eval bbox
```
* You can also refer to [StreamPETR](https://github.com/exiawsh/StreamPETR/blob/main/docs/training_inference.md) for more training recipes.
