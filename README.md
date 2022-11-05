## Kaggle 25th solution for "Google Universal Image Embedding(GUIE)"
* kaggle competition, GUIE 25th solution code
* For the details of the competition, please check this page -> [Google universal image embedding](https://www.kaggle.com/competitions/google-universal-image-embedding/overview)

## Overview
Initially I was working on this competition with text-image contrastive method and trying dimension reduction technique like UMAP and also trying embedding each category(fashion/package/landmark...) on different discrete spaces, but these approach were not good.
The approach of [motono0223's baseline](https://www.kaggle.com/code/motono0223/guie-clip-tensorflow-train-example) was best for me. freezed CLIP + Arcface head.

### Downsampling for the number of classes
* 64D embedding space is not large so I tuned the number of classes for training.
* Training 4000 classes from Products10k/GLR was the best for me.
![image](https://user-images.githubusercontent.com/61892693/195221946-f85f5e92-b57f-4315-bee9-4c9956d71a09.png)

### Hard to create good CV-LB correlation
* I configured 6 different retrieval tasks, Products10k/GLR/Stanford Online Products/DeepFashion/MET/Food-101/ObjectNet but could not find clear CV-LB correlation.
* When only training with Products10k/GLR, Stanford Online Products(Cabinet/Sofa/Chair) retrieval setting was relatively correlated to LB. But it was still not perfect one.
![image](https://user-images.githubusercontent.com/61892693/195222493-90ad0fef-b80a-46b2-bf9c-2e8f57028a3c.png)

### Freeze CLIP
* Unfreeze CLIP finetune does not worked for me. Lowering learning rate got worked but freeze & high learning rate was better.
* Finetuning other backbones, like Imagenet pretrained models, were not good.
* Adding more transformer layers on freezed CLIP was not good.


## Top scripts description

```bash
├ Dockerfile
├ LICENSE
├ pyproject.toml
├ README.md
├ requirements_dev.txt                         # python deps, for development
├ requirements.txt                             # main python deps for this project
├ train_multi_domain.py                        # training
├ eval_multi_domain_with_retrieval.py          # evaluation with retrieval for 6 different datasets
└ setup.cfg
```

## How to run
* This is a competition code and some part is not so clean.
* And due to the nature of this competition, 6~13 different public datasets, Products10k/GLR/Stanford Online Products ... are needed.

### environment
* Ubuntu 18.04
* Python with Anaconda
* NVIDIA GPUx1

### Install dependencies,

```bash
# clone project
$PROJECT=kaggle_google_universal_image_embed
$CONDA_NAME=guie
git clone https://github.com/Fkaneko/$PROJECT

# install project
cd $PROJECT
conda env create -f  ./conda_env.yaml
conda activate $CONDA_NAME
pip install -U pip
pip install -r ./requirements_dev.txt
pip install -r ./requirements.txt
```

and need the following directory configuration
```bash
    ├ input/            # dataset directory
    ├ working/           # training result will be stored here
    └ kaggle_kaggle_google_universal_image_embed/  # this github project.
```
#### Docker

```bash
IMAGE_NAME="kaggle/guie"
TAG="0.0.1"
WORK_DIR_NAME="google_universal_image_embed"

# docker image build. it takes few minutes.
docker build -f Dockerfile . -t ${IMAGE_NAME}:${TAG}

# Start a docker container
wandb docker-run -it --rm \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${HOME}/kaggle/input":"${HOME}/kaggle/input" \
    -v "${HOME}/kaggle/working":"${HOME}/kaggle/working" \
    -v "${HOME}/kaggle/${WORK_DIR_NAME}":"${HOME}/kaggle/${WORK_DIR_NAME}" \
    -w "${HOME}/kaggle/${WORK_DIR_NAME}" \
    ${IMAGE_NAME}:${TAG}

```

### training
Run training with Wandb,
 ```bash
python ./train_multi_domain.py
```
### evaluation
 ```bash
python ./eval_multi_domain_with_retrieval.py
```

## License
* code: Apache 2.0
* dataset used in this project: Please check for each dataset license.

## Reference
