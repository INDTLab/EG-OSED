<p align="center"> <a href="" ><img src="imgs/abstract.png"></a></p>
<p align="center"> 
<a href="" ><img src="https://img.shields.io/badge/HOME-KBS-blue.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/HOME-Paper-important.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/PDF-Paper-blueviolet.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/-Poster-ff69b7.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/-Video-brightgreen.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/-Supplementary-green.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/-WeightsFiles-blue.svg"></a>
</p>

# Architecture

![archioverall](imgs/EG-OSED-YOLOv4.JPG)

# Usage
## Installation
1. Create the environment from the environment.yml file:  
   ```conda env create -f environment.yml```    
2. Activate the new environment:  
   `conda activate py37`  
3. Verify that the new environment was installed correctly:  
    `conda env list`

## Configuration
You can use command like this:  
`python train.py -g gpu_num -subdivision sub_num -name 'proj-name -dataset 'dataset-name' -backbone 'backbone-name' -pretrained 'pretrained-weight' `  
You can also change parameters in the train.py and then :  
`python train.py`  
  
# Data Sets
You can download the data sets uesd in the paper from : https://pan.baidu.com/s/1jmkR3__ONSWmivjZj46nHw?pwd=bl3a 提取码：bl3a    

Then you can get labels file with the formate used in this repo by voc.py as:  
`python voc.py`  


# Citation
