<p align="center"> <a href="" ><img src="imgs/title.JPG"></a></p>
<p align="center"> 
<a href="" ><img src="https://img.shields.io/badge/HOME-KBS-blue.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/HOME-Paper-important.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/PDF-Paper-blueviolet.svg"></a>
<!-- <a href="" ><img src="https://img.shields.io/badge/-Poster-ff69b7.svg"></a> -->
<!-- <a href="" ><img src="https://img.shields.io/badge/-Video-brightgreen.svg"></a> -->
<a href="https://drive.google.com/file/d/1YCEIvquBG44w44tzoDAxpvnQpgddF1CV/view?usp=drive_link" ><img src="https://img.shields.io/badge/-WeightsFiles-blue.svg"></a>
</p>

# Architecture

![archioverall](imgs/EG-OSED-YOLOv4.JPG)

# Usage
### Installation
1. Create the environment from the `environment.yml` file:   
   ```copy
   conda env create -f environment.yml
   ```     
2. Activate the new environment:  
   ```copy
   conda activate py37
   ```    
3. Verify that the new environment was installed correctly:  
    ```copy
   conda env list
    ```    

### Configuration

### Data Sets
Download data sets : <a href="https://drive.google.com/file/d/1D00quOYefmW_VoBnJVNjkezOa2w-aUWl/view?usp=drive_link">GoogleLink</a> or <a href="https://pan.baidu.com/s/1jmkR3__ONSWmivjZj46nHw?pwd=bl3a">BaiduLink</a> with code `bl3a`      

The data sets were annotated in VOC formate and can be converted into the format used in this repo by `voc.py` as:  
```copy
python voc.py
```

#### Train
Download ResNet50 pre-trained on the Places365 data set: <a href="https://drive.google.com/file/d/1Zp251rEyZ-UkuuJ7r206_RyPfIlr9sGa/view?usp=drive_link">GoogleLink</a> or <a href="https://pan.baidu.com/s/1hEaCcMFiAIlcuJdcTPiueg?pwd=e0e6 ">BaiduLink</a>  with code e0e6.
You can use command like this:  
```copy
python train.py -g gpu_num -subdivision sub_num -name 'proj-name' -dataset 'dataset-name' -backbone 'backbone-name' -pretrained 'pretrained-weight'
```  
You can also change parameters in the train.py and then :  
```copy
python train.py
```

#### Test
Evaluates the model on the test data sets.
```copy
python test.py -g gpu_num -name 'proj-name' -dataset 'dataset-name' -w 'weights' -backbone 'backbone-name'
```
  

 

# Result
Quantitative Results on the FOSD_OD, Places365_OD, SUN_OD data sets.  
<div align=center><img src="imgs/Table.PNG" width=50%></div>  
Qualitative Results derived using EG-OSED-YOLOv4 on the FOSD_OD, Places365_OD, SUN_OD data sets.   
<div align=center><img src="imgs/results.JPG"></div>  

# Citation
