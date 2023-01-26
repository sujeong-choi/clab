# Picture Frame Detection (PFD)
## Training
To train the PFD model, take the following steps: 
* Make sure you have the training dataset that can be found on [this url](https://drive.google.com/file/d/1uO0sfcbUmUK9zsLteSXjgR2pomfNUGCo/view?usp=sharing). Extract the zip file to `assets/` folder. 

* Once you extract the dataset, open [keypoint_rcnn](./keypoint_rcnn.ipynb) and follow the instructions there. Make sure you have the following python dependencies installed:
    ```
    numpy==1.22.1
    torch==1.13.1+cu116
    torchvision==0.14.1+cu116
    matplotlib==3.5.1
    pycocotools==2.0.6
    opencv-python==4.6.0.66
    albumentations==1.3.0
    ```

* Run each block of code and train the model
* Last block runs individual tests on images

## Testing
To test the PFD model, take the following steps:
* You'll need a trained model in `assets/keypoint_model/weights`. You can either go through the training phase above by running [keypoint_rcnn](./keypoint_rcnn.ipynb) or by extracting trained models from [this url](https://drive.google.com/file/d/1ylTbpvp5ASnDPY6MoQa6OlgKBzV8y-Uk/view?usp=sharing) into the `assets/` folder. 

* You can add a test video file to `assets/pfd_video_dataset`. The default test video is `20221014_124610.mp4`. You'll have to modify it if you want to test on another video file. 

* All other instructions are provided in the [keypoint_rcnn_inference](./keypoint_rcnn_inference.ipynb) notebook