<h2 align="center"> 
  <!-- <a href="https://arxiv.org/abs/2411.10504"> -->
  Dynamic Prediction of Reynolds-averaged Flow Field past Airfoil at varying Angle of Attack via Physical-guided Video Diffusion Model
</h2>
<h5 align="center"> 
Yifeng Ai*, Yijia Guo*, Peng Guo, Hongfeng Zhang✉, C.W.Lim✉, Tiejun Huang, Lei Ma✉
 </h5>
<h5 align="center"> 
* Equal contributions,
✉ Corresponding authors </a>
</h5>
<!-- <h5 align="center">

<!-- [![arXiv](https://img.shields.io/badge/Arxiv-2411.10504-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.10504)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/chenkang455/USP-Gaussian)
[![GitHub repo stars](https://img.shields.io/github/stars/chenkang455/USP-Gaussian?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/chenkang455/USP-Gaussian/stargazers)&#160; 

</h5> -->
## 👀 Overview
<p align="center">
  <img src="asset/pipeline.png" width="800"/>
</p>



## 📕 Abstract
> We propose a synergistic optimization framework USP-Gaussian, that unifies spike-based image reconstruction, pose correction, and Gaussian splatting into an end-to-end framework. Leveraging the multi-view consistency afforded by 3DGS and the motion capture capability of the spike camera, our framework enables a joint iterative optimization that seamlessly integrates information between the spike-to-image network and 3DGS. Experiments on synthetic datasets with accurate poses demonstrate that our method surpasses previous approaches by effectively eliminating cascading errors. Moreover, we integrate pose optimization to achieve robust 3D reconstruction in real-world scenarios with inaccurate initial poses, outperforming alternative methods by effectively reducing noise and preserving fine texture details.


<!-- ## 👀 Visual Comparisons
<details open>
<summary><strong>Novel-view synthesis comparison on the real-world dataset.</strong></summary>
<p align="center">
<img src="imgs/outdoor.gif" width="49%" height="auto"/>
<img src="imgs/keyboard.gif" width="49%" height="auto"/>
</p>
</details>

<details open>
<summary><strong>Jointly optimized 3DGS and Recon-Net reconstruction on the synthetic dataset.</strong></summary>
<p align="center">
<img src="imgs/tanabata.gif" width="99%" height="auto"/>
</p>
</details> -->


## 🗓️ TODO
- [x] Release the training code.
- [x] Release the dataset.
- [x] Release the pretrained model.
- [ ] Release the data generation code.
- [ ] Release the project page.


## 🕶 Get Started
### 1. Installation
## install torch
```
pip install --upgrade pip setuptools
pip install "torch==2.1.2+cu118" "torchvision==0.16.2+cu118" --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```
## install dependencies
```
conda env create -f environment.yml
```

### 2. Dataset Preparation
Well-organized datasets can be found in the [download link](https://huggingface.co/datasets/diff-flow/diff-flow).

Overall, the structure of our project is formulated as:
```
<project root>
├── data
│   ├── fluid
|   |     ├──p_npy
|   |     ├──u_npy
|   |     ├──v_npy
|   |     └──Token.csv
│   └── target_response.csv
└── main.py
``` 

### 3. Pretrained Model
Pretrained model can be found in the [download link](https://huggingface.co/diff-flow/diff-flow).
Pretrained model should be formulated as:
```
<project root>
├── runs
│   └──Pretrained
|         └──model
|             └──step_100000
|                     └──checkpoint.pt
└── main.py
``` 
<!-- For a comprehensive guide on synthesizing the entire synthetic dataset from scratch, as well as the pose estimation method, please refer to the [Dataset](scripts/Dataset.md) file.  -->

### 3. Training
```
accelerate launch main_our_npy.py --name <you_name> --mode train
```


### 4. Rendering
For rendering 3D scene from the input camera trajectory, run:
```
accelerate launch main_our_npy.py --name pretrained --mode eval
```

## 🙇‍ Acknowledgment
Our code is implemented based on the [Inverse design of nonlinear mechanical metamaterials via video denoising diffusion models](https://www.nature.com/articles/s42256-023-00762-x) by [Phil Wang](https://github.com/jhbastek/VideoMetamaterials) and the video denoising diffusion implementation by [Phil Wang](https://github.com/lucidrains/imagen-pytorch) proposed in [Imagen Video](https://imagen.research.google/video/).

## 🤝 Citation
If you find our work useful in your research, please cite:
```
<!-- @article{chen2024usp,
  title={USP-Gaussian: Unifying Spike-based Image Reconstruction, Pose Correction and Gaussian Splatting},
  author={Chen, Kang and Zhang, Jiyuan and Hao, Zecheng and Zheng, Yajing and Huang, Tiejun and Yu, Zhaofei},
  journal={arXiv preprint arXiv:2411.10504},
  year={2024}
} -->
```
