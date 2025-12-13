# Register Token Merging: Fast, Stable ViT

Official PyTorch implemention of **RegToMe** from our paper: [Register Token Merging: Register Token Merging: Fast, Stable ViT]().  
Anh LE Hai

## What is RegToMe?

Register Token Merging (RegTome) allows you to take an existing Vision Transformer architecture and efficiently merge tokens inside of the network for **2-3x** faster evaluation . RegToMe is tuned to seamlessly fit inside existing vision transformers, so you can use it without having to do additional training. And if you *do* use RegTome during training, you can reduce the accuracy drop even further while also speeding up training considerably.

## What RegToMe does

RegTome merges tokens based on their similarity, implicitly grouping parts of objects together. This is in contrast to token pruning, which only removes background tokens. RegTome can get away with reducing more tokens because we can merge redundant foreground tokens in addition to background ones. Moreover, we can add Register Tokens for better stablization anf reduce artifacts.  Visualization of merged tokens on Cifar10 val. For more, see the paper appendix.


## News
 - **[2025.12.12]** Initial release.

## Installation
See [INSTALL.md](INSTALL.md) for installation details.

## Support Models

This repo does not include training code. Instead, we provide a set of tools to patch existing vision transformer implementations. Then, you can use those implementations out of the box. Currently, we support the following ViT implementations:
 - [x] [🔗](#using-timm-models) [timm](https://github.com/rwightman/pytorch-image-models)
 
**Note:** these external implementations aren't associated with Meta in any way.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/HaiAnh-X/Registers-Merging.git
    cd Registers-Merging
    ```

2. **Create Env**:
 
    ```
    conda env create -f environment.yml
    ```


3. **Prepare the dataset**:
    Ensure that the CIFAR-10 or CIFAR-100 dataset is available in the `./data/CIFAR10` or `./data/CIFAR100` directory respectively. If you want to train on the ImageNet 1k dataset, download the data through [ImageNet](https://www.image-net.org/)

4. **Train the model**:
    Train the Vision Transformer model on the CIFAR-10, CIFAR-100, or ImageNet1k dataset. The model prepared to train on the ImageNet1k has dynamic token locations you can set and returns the evaluation as well after training.
    More details on the training engine are in ```cifar_train.py```
    ```bash
    python main.py
    ```
   

6. **Evaluate the model**:
    Evaluate the trained model on the CIFAR-10 or CIFAR-100 test set.
    More details on the training engine are in ```cifar_test.py```
    ```bash
    python main.py
    ```

Here are some expected results when using the timm implementation *off-the-shelf* on ImageNet-1k val using a V100:

| Model          | original acc | original im/s |  r | RegTome acc | RegTome im/s | Improvement
|----------------|-------------:|--------------:|:--------:|---------:|----------:|----------:|
| ViT-S/16       |       76.81% |        571.25 | 16       |    74.23%|    997.39 |1.75x|
| ViT-S/16       |       76.81% |        571.25 | (16, -1) |    69.43% |      1414.78 | 2.48x|



See the paper for full results with all models and all values of `r`.




