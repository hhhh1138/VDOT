# VDOT: Efficient Unified Video Creation via Optimal Transport Distillation
<a href="https://arxiv.org/"><img src='https://img.shields.io/badge/VDOT-arXiv-red' alt='Paper PDF'></a>
<a href="https://vdot-page.github.io/"><img src='https://img.shields.io/badge/VACE-Project_Page-green' alt='Project Page'></a>
<!-- <a href="https://huggingface.co/collections/ali-vilab/vace-67eca186ff3e3564726aff38"><img src='https://img.shields.io/badge/VACE-HuggingFace_Model-yellow'></a>
<a href="https://modelscope.cn/collections/VACE-8fa5fcfd386e43"><img src='https://img.shields.io/badge/VACE-ModelScope_Model-purple'></a> -->


[Yutong Wang](https://yutongwang1012.github.io/)<sup>1</sup>, [Haiyu Zhang](https://github.com/aejion)<sup>3,2</sup>, [Tianfan Xue](https://tianfan.info/)<sup>4,2</sup>, [Yu Qiao](https://mmlab.siat.ac.cn/yuqiao/)<sup>2</sup>, [Yaohui Wang](https://wyhsirius.github.io/)<sup>2</sup>, [Chang Xu](http://changxu.xyz/)<sup>1*</sup>, [Xinyuan Chen](https://scholar.google.com/citations?user=3fWSC8YAAAAJ&hl=zh-CN)<sup>2*</sup>

<sup>1</sup>USYD, <sup>2</sup>Shanghai AI Laboratory, <sup>3</sup>BUAA, <sup>4</sup>CUHK

## Introduction
<strong>VDOT</strong> is an efficient, unified video creation model that achieves high-quality results in just 4 denoising steps. 
By employing Computational Optimal Transport (OT) within the distillation process, VDOT ensures training stability and enhances both training and inference efficiency.
VDOT unifies a wide range of capabilities, such as <strong>Reference-to-Video (R2V)</strong>, <strong>Video-to-Video (V2V)</strong>, <strong>Masked Video Editing (MV2V)</strong>, and arbitrary <strong>composite tasks</strong>, matching the versatility of VACE with significantly reduced inference costs.

<video autoplay muted loop playsinline controls src="./assets/videos/sour_cover2.jpg">


## ⚙️ Installation
The codebase was tested with Python 3.10.13, CUDA version 12.4, and PyTorch >= 2.5.1.


## 🚀 Usage


## Acknowledgement
We are grateful for the following awesome projects, including [VACE](https://github.com/ali-vilab/VACE), [Wan](https://github.com/Wan-Video/Wan2.1), and [Self-Forcing](https://github.com/guandeh17/Self-Forcing).


## BibTeX

```bibtex
