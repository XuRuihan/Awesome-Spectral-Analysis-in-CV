# 🚀 Awesome Spectral Analysis in CV

<!-- Badges: license, CI, stars -->
[![License: Apache](https://img.shields.io/badge/License-Apache-yellow.svg)](https://opensource.org/license/apache-2-0)
[![GitHub stars](https://img.shields.io/github/stars/XuRuihan/Awesome-Spectral-Analysis-in-CV.svg)](https://github.com/XuRuihan/Awesome-Spectral-Analysis-in-CV/stargazers)
<!-- ![Papers](https://img.shields.io/badge/Papers-100-blue) -->

🔥 This projection collects and organizes high-quality  resources, papers, and codes applying **Spectral Analysis** (Frequency Domain & Time-Frequency Analysis) in Computer Vision.

✨ Key Features
* 🎯 Comprehensive Coverage: Model Architecture, Learning Strategy, and Visual Applications.
* 🏢 Multi-domain Applications: Generative Models, Deepfake Detection, Low-Level Vision, 3D-Vision & Neural Rendering, etc.
* 📊 Systematic Organization: Well-structured taxonomy and unified workflows
* 🔄 Regular Updates: Keep up with the latest research developments

## 📖 Introduction

While Convolutional Neural Networks (CNNs) and Transformers excel in the spatial domain, the **Spectral Domain** offers unique advantages for modern Computer Vision challenges. By leveraging tools like the **Fast Fourier Transform (FFT)**, **Discrete Cosine Transform (DCT)**, and **Discrete Wavelet Transform (DWT)**, researchers are addressing critical issues such as:

*   **Efficiency:** Replacing $O(N^2)$ self-attention with $O(N \log N)$ FFT operations.
*   **Global Receptive Fields:** Capturing long-range dependencies inherently in the spectral domain.
*   **Domain Generalization:** Disentangling style (Amplitude) from content (Phase) for robust transfer learning.
*   **Generative Quality:** Fixing "checkerboard artifacts" and spectral bias in GANs/Diffusion models.


## 📋 Table of Contents

- [🚀 Awesome Spectral Analysis in CV](#🚀-awesome-spectral-analysis-in-cv)
  - [📖 Introduction](#📖-introduction)
  - [📋 Table of Contents](#📋-table-of-contents)
  - [1. Model Architecture](#1-model-architecture)
    - [1.1. Efficient Architectures](#11-efficient-architectures)
  - [2. Learning Strategy](#2-learning-strategy)
    - [2.1. Domain Adaptation \& Generalization](#21-domain-adaptation--generalization)
  - [3. Visual Application](#3-visual-application)
    - [3.1 Generative Models \& Synthesis](#31-generative-models--synthesis)
    - [3.2 Deepfake Detection \& Forensics](#32-deepfake-detection--forensics)
    - [3.3 Low-Level Vision (SR, Compression)](#33-low-level-vision-sr-compression)
    - [3.4. 3D Vision \& Neural Rendering](#34-3d-vision--neural-rendering)
  - [📚 Tutorials \& Tools](#📚-tutorials--tools)
  - [🤝 Contribution](#🤝-contribution)
  - [⚖️ License](#⚖️-license)


## 1. Model Architecture

### 1.1. Efficient Architectures

*Leveraging spectral transformations to replace or enhance token mixing mechanisms for faster inference and global context.*

| Paper Title | Conference | Description | Code |
| :--- | :---: | :--- | :---: |
| [FNet: Mixing Tokens with Fourier Transforms](https://aclanthology.org/2022.naacl-main.319/) | NAACL 2021 | Replaces the self-attention layer in Transformers with a standard unparameterized Fourier Transform. Achieves 92-97% of BERT accuracy but trains significantly faster. | [![GitHub stars](https://img.shields.io/github/stars/google-research/google-research.svg)](https://github.com/google-research/google-research/tree/master/f_net)|
| [Global Filter Networks for Image Classification](https://proceedings.neurips.cc/paper/2021/hash/07e87c2f4fc7f7c96116d8e2a92790f5-Abstract.html) | NeurIPS 2021 | GFNet learns global filters in the frequency domain, which obtains a global receptive field without stacking layers. It performs FFT, multiplies with a learnable filter, and performs IFFT, achieving global receptive fields without deep stacking. | [![GitHub stars](https://img.shields.io/github/stars/raoyongming/GFNet.svg)](https://github.com/raoyongming/GFNet) |
| [Adaptive Frequency Filters As Efficient Global Token Mixers](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_Adaptive_Frequency_Filters_As_Efficient_Global_Token_Mixers_ICCV_2023_paper.html) | ICCV 2023 | Proposes a lightweight module to dynamically emphasize discriminative frequency components within an efficient CNN architecture. | [![GitHub stars](https://img.shields.io/github/stars/sunpro108/AdaptiveFrequencyFilters.svg)](https://github.com/sunpro108/AdaptiveFrequencyFilters) |


## 2. Learning Strategy

### 2.1. Domain Adaptation & Generalization

*Using Amplitude and Phase spectra to separate style from semantic content.*

| Paper Title | Conference | Description | Code |
| :--- | :---: | :--- | :---: |
| [FDA: Fourier Domain Adaptation for Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html) | CVPR 2020 | Performs unsupervised domain adaptation by swapping the low-frequency amplitude spectrum of source images with target images. | [![GitHub stars](https://img.shields.io/github/stars/YanchaoYang/FDA.svg)](https://github.com/YanchaoYang/FDA) |
| [FSDR: Frequency Space Domain Randomization for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_FSDR_Frequency_Space_Domain_Randomization_for_Domain_Generalization_CVPR_2021_paper.html) | CVPR 2021 | Randomizes frequency components during training to force the model to learn structure (phase) rather than texture (amplitude), improving generalization. | [![GitHub stars](https://img.shields.io/github/stars/jxhuang0508/FSDR.svg)](https://github.com/jxhuang0508/FSDR/stargazers) |
| [FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.html) | CVPR 2021 | Uses frequency space interpolation to share distribution information across federated clients without leaking private data. | [![GitHub stars](https://img.shields.io/github/stars/liuquande/FedDG-ELCFS.svg)](https://github.com/liuquande/FedDG-ELCFS) |
| [Amplitude-Phase Recombination: Rethinking Robustness of Convolutional Neural Networks in Frequency Domain](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Amplitude-Phase_Recombination_Rethinking_Robustness_of_Convolutional_Neural_Networks_in_Frequency_ICCV_2021_paper.html) | ICCV 2021 | CNN tends to converge at the local optimum which is closely related to the high-frequency components of the training images, while the amplitude spectrum is easily disturbed such as noises or common corruptions. | [![GitHub stars](https://img.shields.io/github/stars/iCGY96/APR.svg)](https://github.com/iCGY96/APR) |


## 3. Visual Application

### 3.1 Generative Models & Synthesis

*Addressing spectral artifacts and accelerating sampling in GANs and Diffusion Models.*

| Paper Title | Conference | Description | Code |
| :--- | :---: | :--- | :---: |
| [Wavelet Diffusion Models are fast and scalable Image Generators](https://openaccess.thecvf.com/content/CVPR2023/html/Phung_Wavelet_Diffusion_Models_Are_Fast_and_Scalable_Image_Generators_CVPR_2023_paper.html) | CVPR 2023 | Incorporates wavelet transforms into diffusion models to perform denoising in the frequency domain, enabling faster convergence and high-frequency preservation. | [![GitHub stars](https://img.shields.io/github/stars/VinAIResearch/WaveDiff.svg)](https://github.com/VinAIResearch/WaveDiff) |
| [Focal Frequency Loss for Image Reconstruction and Synthesis](http://openaccess.thecvf.com/content/ICCV2021/html/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.html) | ICCV 2021 | Introduces a loss function that optimizes the distance between real and generated images in the frequency domain to remove spatial artifacts. | [![GitHub stars](https://img.shields.io/github/stars/EndlessSora/focal-frequency-loss.svg)](https://github.com/EndlessSora/focal-frequency-loss) |
| [On the Spectral Bias of Neural Networks](https://proceedings.mlr.press/v97/rahaman19a.html) | ICML 2019 | Explains why neural networks tend to learn low-frequency functions first and struggle with high-frequency details. | [![GitHub stars](https://img.shields.io/github/stars/nasimrahaman/SpectralBias.svg)](https://github.com/nasimrahaman/SpectralBias) |



### 3.2 Deepfake Detection & Forensics

*Detecting invisible fingerprints left by up-sampling operations on the spectral domain.*

| Paper Title | Conference | Description | Code |
| :--- | :---: | :--- | :---: |
| [CNN-generated images are surprisingly easy to spot... for now](http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html) | CVPR 2020 | Demonstrates that GANs leave specific fingerprints in the frequency spectrum that allow for easy detection of fake images. | [![GitHub stars](https://img.shields.io/github/stars/PeterWang512/CNNDetection.svg)](https://github.com/PeterWang512/CNNDetection) |
| [Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Learning](https://ojs.aaai.org/index.php/AAAI/article/view/28310) | AAAI 2024 | A multi-stream network that specifically mines frequency-level inconsistencies to detect face swapping. | [![GitHub stars](https://img.shields.io/github/stars/chuangchuangtan/FreqNet-DeepfakeDetection.svg)](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) |
| [Spatial-Phase Shallow Learning: Rethinking Face Forgery Detection in Frequency Domain](http://openaccess.thecvf.com/content/CVPR2021/html/Liu_Spatial-Phase_Shallow_Learning_Rethinking_Face_Forgery_Detection_in_Frequency_Domain_CVPR_2021_paper.html) | CVPR 2021 | Capture the up-sampling artifacts of face forgery | - |


### 3.3 Low-Level Vision (SR, Compression)

*Super-Resolution, Denoising, and Image Compression.*

| Paper Title | Conference | Description | Code |
| :--- | :---: | :--- | :---: |
| [Deep Wavelet Prediction for Image Super-Resolution](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Guo_Deep_Wavelet_Prediction_CVPR_2017_paper.html) | CVPR 2017 | Utilizes wavelet decomposition to recover missing high-frequency details in sub-bands for cleaner super-resolution. | [![GitHub stars](https://img.shields.io/github/stars/tT0NG/DWSRx3.svg)](https://github.com/tT0NG/DWSRx3) |


### 3.4. 3D Vision & Neural Rendering

*Overcoming spectral bias to render high-fidelity 3D scenes.*

| Paper Title | Conference | Description | Code |
| :--- | :---: | :--- | :---: |
| [NeRF: Representing Scenes as Neural Radiance Fields](https://dl.acm.org/doi/abs/10.1145/3503250) | ECCV 2020 | Shows that mapping input coordinates to higher dimensional frequency bands (Positional Encoding) is crucial for rendering fine details. | [![GitHub stars](https://img.shields.io/github/stars/bmild/nerf.svg)](https://github.com/bmild/nerf) |
| [Fourier Features Let Networks Learn High Frequency Functions](https://proceedings.neurips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html) | NeurIPS 2020 | Provides the theoretical backing for why Fourier feature mapping enables MLPs to learn high-frequency content in low-dimensional domains. | [![GitHub stars](https://img.shields.io/github/stars/tancik/fourier-feature-networks.svg)](https://github.com/tancik/fourier-feature-networks) |


## 📚 Tutorials & Tools

If you are new to Signal Processing in Deep Learning, start here:

*   **Textbooks**: The basic theorems for spectral domain analysis could be found in any textbooks about signal processing and digital image processing.
*   **Docs:** [PyTorch FFT Documentation](https://pytorch.org/docs/stable/fft.html)
*   **Library:** [PyWavelets - Wavelet Transform in Python](https://pywavelets.readthedocs.io/en/latest/)


## 🤝 Contribution

Contributions are welcome! If you have a paper or project to add, please follow these steps:

1.  Fork this repository.
2.  Create a branch: `git checkout -b feature/AddPaperName`.
3.  Add the paper to the appropriate category table.
4.  Commit your changes and push to the branch.
5.  Push to the branch and submit a Pull Request.

Please ensure the format follows: `| [Paper Title](Paper Link) | Conference | Description | [Github stars](Github Link) |`


## ⚖️ License

Distributed under the Apache-2.0 License. See `LICENSE` for more information.
