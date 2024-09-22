# Restorer: Removing Multi-Degradation with All-Axis Attention and Prompt Guidance
<a href="https://arxiv.org/abs/2406.12587"><img src="https://img.shields.io/badge/arXiv-2406.12587-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>  
 <a href="https://arxiv.org/abs/2406.12587"> </a> 

Code for the paper [Restorer: Removing Multi-Degradation with All-Axis Attention and Prompt Guidance](https://arxiv.org/abs/2406.12587)

Note: This code base is still not complete. 

[Paper](https://arxiv.org/abs/2406.12587) 

### About this repo:

This repo hosts the implentation code for the paper "Restorer".

## Introduction

> There are many excellent solutions in image restoration. However, most methods require on training separate models to restore images with different types of degradation. Although existing all-in-one models effectively address multiple types of degradation simultaneously, their performance in real-world scenarios is still constrained by the task confusion problem. In this work, we attempt to address this issue by introducing **Restorer**, a novel Transformer-based all-in-one image restoration model. To effectively address the complex degradation present in real-world images, we propose All-Axis Attention (AAA), a novel attention mechanism that simultaneously models long-range dependencies across both spatial and channel dimensions, capturing potential correlations along all axes. Additionally, we introduce textual prompts in Restorer to incorporate explicit task priors, enabling the removal of specific degradation types based on user instructions. By iterating over these prompts, Restorer can handle composite degradation in real-world scenarios without requiring additional training. Based on these designs, Restorer with one set of parameters demonstrates state-of-the-art performance in multiple image restoration tasks compared to existing all-in-one and even single-task models. Additionally, Restorer is efficient during inference, suggesting the potential in real-world applications.


<div align=center>
<img src="./\imgs\pipeline.png" width="1000"/>
</div>

## News 🚀
* **2024.09.21**: Code and checkpoint are released.
* **2024.09.03**: [Paper](https://arxiv.org/abs/2406.12587) is released on ArXiv.

## Quick Start

### Install

- python 3.7
- cuda >=10.1

```
# git clone this repository
git clone https://github.com/Talented-Q/Restorer.git
cd Restorer

# create new anaconda env
conda create -n Restorer python=3.7
conda activate Restorer 

# install packages
pip install -r requirements.txt
```


## Datasets:

### Dataset setting:

| Task | Dataset | #Train | #Test | Test Dubname |
|:------------------------------------------------------------------:|:-------------:|:----------:|:----------:|:----------:|
| Desnowing    |       [CSD](https://ccncuedutw-my.sharepoint.com/:u:/g/personal/104501531_cc_ncu_edu_tw/EfCooq0sZxxNkB7F8HgCyKwB-sJQtVE59_Gpb9soatYi5A?e=5NjDhb)        |   5000   | 2000 | CSD |
| Deraining    |       [RAIN1400](https://drive.google.com/file/d/10cu6MA4fQ2Dz16zfzyrQWDDhOWhRdhpq/view)       |   5000   | 1400 | rain1400 |
| Dehazing    |       [OTS](https://utexas.app.box.com/s/25idwrsn890w03grdr6pls28cy38r91i)       |   5000   | 500 | SOTS |
| Denoising    |       [SIDD](https://abdokamel.github.io/sidd/)        |   5000   | 1280 | SIDD  |
| Debluring    |       [GoPro](https://drive.google.com/file/d/1SlURvdQsokgsoyTosAaELc4zRjQz9T2U/view?usp=sharing)      |   2103   | 1111 | GoPro |
| Debluring    |       [RealBlur-R](https://drive.google.com/file/d/1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS/view?usp=sharing)        |   3758   | 980 | RealBlur-R  |
| Lowlight Enhancement    |       [LOL](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view)        |   485   | 15 | LOL  |

### Train data:

Restorer is trained on a combination of images sampled from CSD, rain1400, OTS datasets (similar to [TKL (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_Multiple_Adverse_Weather_Removal_via_Two-Stage_Knowledge_Learning_and_CVPR_2022_paper.pdf)), SIDD, GoPro, RealBlur-R, and LOL dubbed as "_mixed training set_", containing 26346 images.


### Dataset format:

Please download and sample the corresponding number of datasets according to our dataset setting and arrange them in the following format.
Download the val datasets in this [link](https://drive.google.com/file/d/1gQ27k9O9Wj6Sc4q3lru5SNnHKpHT2d06/view?usp=sharing). 
```
    Restorer
    ├── train 
    |   ├── input # Training  
    |   |   ├── <degradation_kind1>   
    |   |   |   ├── 1.png          
    |   |   |   └── ...    
    |   |   ├── <degradation_kind2> 
    |   |   └── ... 
    |   |
    |   ├── gt # Training  
    |   |   ├── <degradation_kind1>   
    |   |   |   ├── 1.png          
    |   |   |   └── ...    
    |   |   ├── <degradation_kind2> 
    |   |   └── ... 
    |
    ├── val      
    |   ├── <degradation_kind1>          
    |   |   ├── input         
    |   |   |   ├── 1.png          
    |   |   |   └── ...     
    |   |   |── gt         
    |   |   |   ├── 1.png          
    |   |   |   └── ...        
    |   ├── <degradation_kind2>    
    |   └── ... 
```
## Training

### Training the Restorer
```
python train.py --train_root=train_root --val_root=val_root
```

## Evaluating

### Testing the Restorer
Please download our [checkpoint](https://drive.google.com/file/d/1XtX3w0H8EofBep6MN1QYe5Frpa79XTbC/view?usp=drive_link) and put it in `checkpoint path`.
```
python evaluate.py --val_root=val_root --task=task --save=save --ckpt_path=checkpoint path
```

## Performance

### Comparison with unified image restoration methods:
<div align=center>
<img src="./\imgs\Fig6.png" width="1000"/>
</div>

### Comparison with expert networks:
<div align=center>
<img src="./\imgs\Fig12.png" width="1000"/>
</div>

### Real world test:
<div align=center>
<img src="./\imgs\Fig8.png" width="1000"/>
</div>

### Composite degradation restoration:
####lowlight+blur:
<div align=center>
<img src="./\imgs\Figc1.png" width="1000"/>
</div>

####lowlight+noise:
<div align=center>
<img src="./\imgs\Figc2.png" width="1000"/>
</div>

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from [Transweather](https://github.com/jeya-maria-jose/TransWeather), [Syn2Real](https://github.com/rajeevyasarla/Syn2Real), [Segformer](https://github.com/NVlabs/SegFormer), and [ViT](https://github.com/lucidrains/vit-pytorch).

### Citation:

```
@misc{mao2024restorerremovingmultidegradationallaxis,
      title={Restorer: Removing Multi-Degradation with All-Axis Attention and Prompt Guidance}, 
      author={Jiawei Mao and Juncheng Wu and Yuyin Zhou and Xuesong Yin and Yuanqi Chang},
      year={2024},
      eprint={2406.12587},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.12587}, 
}
```
