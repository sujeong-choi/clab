# Picture Frame Detection (PFD)
## Training
To train the PFD model, take the following steps: 
* Make sure you have the training dataset that can be found on [this](https://drive.google.com/file/d/1uO0sfcbUmUK9zsLteSXjgR2pomfNUGCo/view?usp=sharing) url. Extract the zip file to `assets/` folder. 

* Once you extract the dataset, open keypoint_rcnn.ipynb. Make sure you have the following python dependencies installed:
```
opencv-python
torch
albumentations
numpy
matplotlib
```

* Run each block of code and train the model
* Last block runs individual tests on images

## Testing
To run the tests, you'll need a trained model in `assets/keypoint_model/weights` and a test video file in `assets/pfd_video_dataset`. The default test video is `20221014_124610.mp4`. You'll have to modify it if you want to test another video file.