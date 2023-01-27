# Efficient Graph Convolutional Network (EfficientGCN) (ResGCNv2.0)


## 1 Prerequisites

### 1.1 Libraries

This code is based on [Python3](https://www.anaconda.com/) (anaconda, >= 3.5) and [PyTorch](http://pytorch.org/) (>= 1.6.0).

Other Python libraries are presented in the **'scripts/requirements.txt'**, which can be installed by 
```
pip install -r scripts/requirements.txt
```

### 2.2 Experimental Dataset

Our models are experimented on the **NTU RGB+D 60 & 120** datasets, which can be downloaded from 
[here](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).

There are 302 samples of **NTU RGB+D 60** and 532 samples of **NTU RGB+D 120** need to be ignored, which are shown in the **'src/preprocess/ignore.txt'**.


## 2 Running

### 2.1 Modify Configs

Firstly, you should modify the **'path'** parameters in all config files of the **'configs'** folder.

A python file **'scripts/modify_configs.py'** will help you to do this. You need only to change three parameters in this file to your path to NTU datasets.
```
python scripts/modify_configs.py --path <path/to/save/numpy/data> --ntu60_path <path/to/ntu60/dataset> --ntu120_path <path/to/ntu120/dataset> --pretrained_path <path/to/save/pretraiined/model> --work_dir <path/to/work/dir>
```
All the commands above are optional.

### 2.2 Generate Datasets

After modifing the path to datasets, please generate numpy datasets by using **'scripts/auto_gen_data.sh'**.
```
bash scripts/auto_gen_data.sh
```
It may takes you about 2.5 hours, due to the preprocessing module for X-view benchmark (same as [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)).

To save time, you can only generate numpy data for one benchmark by the following command (only the first time to use this benchmark).
```
python main.py -c 2012 -gd
```
**Note:** only training the X-view benchmark requires preprocessing module.

### 2.3 Train

You can simply train the model by 
```
python main.py -c 2012 -g <gpu number>
```
If you want to restart training from the saved checkpoint last time, you can run
```
python main.py -c 2012 -r -g <gpu numbers>
```

### 2.4 Evaluate

Before evaluating, you should ensure that the trained model corresponding the config is already existed in the **<--pretrained_path>** or **'<--work_dir>'** folder. Then run
```
python main.py -c 2012 -e -g <gpu numbers>
```

### 2.5 Visualization

To visualize the details of the trained model, you can run
```
python main.py -c <config> -ex -v -g <gpu numbers>
```
where **'-ex'** can be removed if the data file **'extraction_`<config>`.npz'** already exists in the **'./visualization'** folder.

### 2.6 Runner

To process .mp4 videos by using trained model, you can run
```
python main.py -c 2012 -run -vp <video foleder path> --fps 30 -g <gpu numbers>
```
Runner class infer one label for videos of approximately 2-3 seconds.
The inference result is stored in the video/out folder, and the label displayed in the result video means top 1 to 5 from the top.
