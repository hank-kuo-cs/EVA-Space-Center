# EVA-Space-Center
## Introduction
- This is a ball coordinate regression of the moon.
- Data consists of 100,000 moon images from random angles and distance.
- And the label(ground truth) is 3 parameters: `gamma`, `phi`, `theta`
- The ranges of this 3 parameters list below:
    - `gamma`: [1.7438903314, 1.7579369712]
    - `phi`: [0, 2pi]
    - `theta`: [0, pi]
- The goal of this experiment is to predict the position of the camera in the ball coordinate based on one Moon image.
## Config
> Before training or testing you have to adjust config.py.
- Basic Setting
    - change your logging and CUDA GPU setting.
- Path
    - `DATASET_PATH`: You must set your own dataset path.
    - `WRITER_PATH` & `EXPERIMENT_NAME`: Set the path and the name of the tensorboard.
- Dataset
    - You can adjust your own split way on the train, test, validation set.
    But advise you not change this setting, maybe a bug will happen when loading data.
- Loss Function
    - Only you can change is `GAMMA_WEIGHT`. It means the weight of gamma loss compared to phi and theta.
- Visualization
    - `LOG_STEP`: How many steps to record the loss on the tensorboard and print on the console.
- Hyperparameters
    - All hyperparameters list in this section. You can change any of them by your own way.

## Network Architecture
- The network model is VGG19 + Global Average Pooling Layer + Fully Connected Network, you can see graph of architecture below. 
- Of course, you can choose your own model, but be aware that input image is of one channel, your model should handle that problem.
<img src="https://github.com/hank-kuo-cs/EVA-Space-Center/blob/master/src/Network_Architecture.png" height="80%" width="80%">

## Custom Loss Function - BCMSE & BCL1
- Because phi & theta are direction of circles, we can't use normal loss function to calculate the distance loss.
- I design custom loss function based on MSE & L1 loss function to handle this issue.
- If the parameter is phi or theta, I will calculate the shortest distance in the circle between predict direction and the target direction. This is an alternation of MSE & L1. They called `Ball Coordinate MSE` (BCMSE) and `Ball Coordinate L1` (BCL1).
- The graph below is an example of calculating BCMSE between a prediction phi and a target phi.
<img src="https://github.com/hank-kuo-cs/EVA-Space-Center/blob/master/src/BCMSE.png" height="50%" width="50%">

## Train
### Data Preprocessing
1. Because the image is 800 x 600, it is really slow down the training time, I use gaussian pyramid down to resize it to 400 x 300.
2. I Load the image in gray scale, since I think the color information is not really important to the position regression.
3. Using histogram equalization to strengthen edges and textures of the moon image. 
### Command Line Usage
- Just quickly start by the command below.
```=bash
python3 train.py
```
- `train.py` will look for the newest model in the `checkpoint` directory to train.
- There are some arguments you can use:
    - `-m model_epochxxx.pth`: Choose a particular pretrained model to continue training.
    - `-s`: Train from scratch.


## Test
## Result


