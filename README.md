# CTNet: Color Transformation Network for Low-light Image Enhancement

This is the official PyTorch code for the paper "[[CTNet: Color Transformation Network for Low-light Image Enhancement]](https://www.sciencedirect.com/science/article/abs/pii/S0031320325010210)". This paper was published in Pattern Recognition.

####  

> **Abstract:** Low-light images are often plagued by low visibility, poor contrast, and high noise levels, which significantly impair both subjective visual quality and the performance of downstream tasks. Existing enhancement methods typically struggle with color-related degradations such as color casting, artifacts, and distortion. To address these challenges, we propose an end-to-end Color Transformation Network for low-light image enhancement, with a specific focus on improving color restoration. By leveraging the complementary strengths of the HSV and RGB color spaces in capturing color attributes, our approach enables effective interaction between these color spaces at the feature level. The HSV branch simultaneously enhances the V component while extracting features from the H and S components, thereby providing a more comprehensive set of cues for color recovery. To facilitate interaction, we design a learnable Color Transformation Block that bridges the HSV and RGB feature domains, effectively simulating the HSV-to-RGB conversion. Furthermore, a Cross-Integration Block, employing an attention-based cross-guidance mechanism, enables bi-directional information flow between the two color spaces. Extensive experiments on both real and synthetic datasets demonstrate that our method achieves superior performance, surpassing existing approaches both qualitatively and quantitatively.


##  Contents

- [x] [Datasets](https://github.com/NJUPT-IPR-XuLintao/UPT-Flow/blob/main/README.md#-datasets)
- [ ] Training
- [x] [Testing](https://github.com/NJUPT-IPR-XuLintao/UPT-Flow/blob/main/README.md#-testing)
- [x] [Results](https://github.com/NJUPT-IPR-XuLintao/UPT-Flow/blob/main/README.md#-results)
- [x] [Acknowledgements](https://github.com/NJUPT-IPR-XuLintao/UPT-Flow/blob/main/README.md#-acknowledgements)

##  Datasets

1、LOLv2 (real & synthetic): Wenhan Yang, Haofeng Huang, Wenjing Wang, Shiqi Wang, and Jiaying Liu. "Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement", TIP, 2021. [[Baiduyun (extracted code: l9xm)]](https://pan.baidu.com/s/1U9ePTfeLlnEbr5dtI1tm5g) [Google Drive](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing) 

2、SMID and SDSD (indoor & outdoor): Please refer to [SNRNet(CVPR2022)](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance)

##  Testing

**Pre-trained models for 6 datasets can be obtained from [Google Cloud Drive](https://drive.google.com/drive/folders/1kc1gYk3oTNkV-wZuqUjcZDNbZXqwq5Np?usp=sharing)**

1、Modify the paths to dataset and pre-trained mode. You need to modify the following path in the config files in `./confs`
```python
#### Test Settings
dataroot_unpaired: 
dataroot_GT: put high-light images  
dataroot_LR: put low-light images
model_path: put pre-trained model
```

2、Test the model

To test the model with paired data and obtain the evaluation results, e.g., PSNR, SSIM, and LPIPS. You need to specify the data path ```dataroot_LR```, ```dataroot_GT```, and model path ```model_path``` in the config file. Then run
```bash
test.py 
```

Note that for the LOL datasets, set the window_size to 5, and for the remaining datasets, set it to 8. Modify [Lowlight_Encoder.py](https://github.com/NJUPT-IPR-XuLintao/UPT-Flow/blob/77f391d6b5eb64b2d702c26a782fd70a71c75af4/UPT-Flow/models/modules/Lowlight_Encoder.py#L727)


##  Results

We achieved state-of-the-art performance on *low-light image enhancement*, *night traffic monitoring enhancement*, *low-light object detection* and *Nighttime semantic segmentation*. More results can be found in the paper.

<details>
<summary>Quantitative Comparison (click to expan)</summary>


  <p align="center">
  <img width="900" src="figs/table1.jpg">
	</p>

  <p align="center">
  <img width="500" src="figs/table2.jpg">

  </details>

<details>
<summary>Visual Comparison (click to expan)</summary>


  <p align="center">
  <img width="900" src="figs/fig1.jpg">
	</p>

  <p align="center">
  <img width="900" src="figs/fig2.jpg">
	</p>

  <p align="center">
  <img width="900" src="figs/fig3.jpg">
	</p>

   <p align="center">
  <img width="900" src="figs/fig4.jpg">
	</p>
 
  <p align="center">
  <img width="900" src="figs/fig5.jpg">
	</p>
 
  </details>

## Contact

If you have any questions, please feel free to contact me via email at xielidong@buaa.edu.cn.

## Citation
If you find our work useful for your research, please cite our paper
```
@article{XIE2026112360,
title = {CTNet: Color transformation network for low-light image enhancement},
journal = {Pattern Recognition},
volume = {172},
pages = {112360},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.112360},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325010210},
author = {Lidong Xie and Runmin Cong and Ju Dai and Wenhan Yang and Junjun Pan and Hao Wu},
}
```

##  Acknowledgements
The codes are based on [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer), [Restormer](https://github.com/swz30/Restormer), and [Uformer](https://github.com/ZhendongWang6/Uformer). Please also follow their licenses. Thanks for their awesome works.


