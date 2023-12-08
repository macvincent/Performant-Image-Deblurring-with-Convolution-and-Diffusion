# Performant Image Deblurring with Convolution and Diffusion
---

> **Abstract:** *Image deblurring is a task with multiple real-world use cases. Convolutional Neural Network-based approaches to this task learn priors that generalize well to large-scale data. However, their performance is negatively impacted by their inability to model long-range pixel dependencies and by the static nature of their weights at inference. Vision transformers solve the long-range dependency issue but are computationally inefficient. Our method utilizes convolution and probabilistic diffusion models to efficiently perform the image deblurring task. We show that our method approaches the SOTA while remaining computationally efficient. 
state-of-the-art methods.*
---

## File Structure
This code is built off the [GitHub code](https://github.com/zhengchen1999/HI-Diff) published by Hi-Diff. These are the files we added or significantly modified for the purpose of testing out our hypothesis.
```
├── hi_diff
│   ├── archs
│   │   ├── PIDCD_arch.py
├── options
│   ├── train
│   │   ├── GoPro_S1_PIDCD.yml
│   │   ├── GoPro_S2_PIDCD.yml
│   ├── test
│   │   ├── GoPro_PIDCD.yml
├── evaluate_gopro_hide.py
```

## Baseline Code
The code we used for our NAFNet baselines can be found in [this repo](https://github.com/macvincent/naafnet_baselines). 

## Installation

- Python 3.9
- PyTorch 1.9.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Clone the github repo and go to the default directory 'pidcd'.
git clone https://github.com/macvincent/pidcd
cd pidcd
conda create -n pidcd python=3.9
conda activate pidcd
pip install -r requirements.txt
```

## Datasets

| Dataset                            |           Description            |                             Link                             |
| ---------------------------------- | :------------------------------: | :----------------------------------------------------------: |
| GoPro                              |        Training + Testing        | [Google Drive](https://drive.google.com/file/d/1KYmgaQj0LWSCL6ygtXcuBZ6DfJgO09RQ/view?usp=drive_link) |

Download training and testing datasets and put them into the corresponding folders of `datasets/`. See [datasets](datasets/README.md) for the detail of the directory structure.

## Models

| Model              | Training Dataset | PSNR (dB) | SSIM  
| ------------------ | :--------------: | :-------: | :---: |  
| PIDCD      |      GoPro       |   27.613   | 0.8424 |

## Training

- Download [GoPro](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link) datasets, place them in `datasets/`.

- Generate image patches from GoPro dataset for training.

  ```python
  python generate_patches_gopro.py 
  ```

- Run the following scripts. The training configuration is in `options/train/`.

  ```shell
  # Synthetic, GoPro, 2 Stages, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train/GoPro_S1_PIDCD.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train/GoPro_S2_PIDCD.yml --launcher pytorch
  ```

- The training experiment is in `experiments/`.

## Testing

- Run the following scripts after the two stages of training. The testing configuration is in `options/test/`.
  ```python
  # generate images
  python test.py -opt options/test/GoPro_PIDCD.yml
  # test PSNR/SSIM
  python evaluate_gopro_hide.py
  ```

## Acknowledgements

This code is built off the [GitHub code](https://github.com/zhengchen1999/HI-Diff) published by Hi-Diff.