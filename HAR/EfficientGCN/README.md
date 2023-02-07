# Efficient Graph Convolutional Network (EfficientGCN) (ResGCNv2.0)


## 1 Prerequisites

### 1.1 Libraries

This code is based on [Python3](https://www.anaconda.com/) (anaconda, >= 3.5) and [PyTorch](http://pytorch.org/) (>= 1.6.0).

Other Python libraries are presented in the **'scripts/requirements.txt'**, which can be installed by 
```
pip install -r scripts/requirements.txt
```

### 1.2 Experimental Dataset

Our models are experimented on the **NTU RGB+D 60 & 120** datasets, which can be downloaded from 
[here](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).

There are 302 samples of **NTU RGB+D 60** and 532 samples of **NTU RGB+D 120** need to be ignored, which are shown in the **'src/preprocess/ignore.txt'**.

#### dataset configuration

* painting 500 * 2 ( augmentation )
* Staring 500 * 2 ( augmentation )
* nothing 900 ( from ntu dataset )

#### classes for nothing
* selection criteria : Actions that can come out except painting and interview(staring) in a drawing environment
(그림그리는 환경에서 painting, interview(staring)를 제외하고 나올 수 있는 행동)

1.  1~60 classes 중 
* A1: drink water ; 물 마시기
* A4: brush hair ; 빗으로 머리 빗기
* A5: drop ; 물건 떨어뜨림
* A6: pick up ; 떨어진 물건 줍기
* A10: clapping ; 손벽 치기
* A12: writing ; 쓰기 ??
* A18: put on glasses ; 안경 쓰기
* A19: take off glasses ; 안경 벗기
* A28: phone call ; 전화하기
* A29: play with phone/tablet ; 핸드폰하기
* A33: check time (from watch) ; 손목시계보기
* A49: fan self ; 손부채질하기

2. 61~120
* A68: flick hair ; 머리 정리(만지기)
* A78: open bottle ; 뚜껑 열기
* A91: open a box ; 박스 열기
* A96: cross arms ; 팔짱 끼기
* A103: yawn ; 하품하기
* A104: stretch oneself ; 스트레칭


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
