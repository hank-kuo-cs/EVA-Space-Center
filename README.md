# EVA-Space-Center
## Introduction
- This is a ball coordinate regression of the moon.
- Data consists of 100,000 moon images from random angles and distance.
- And the label(ground truth) is 5 parameters: `c_gamma`, `c_theta`, `c_phi`, `p_theta`, `p_phi`
- The ranges of this 5 parameters list below:
    - c is about the camera, and p is about the optical axis (which direction the camera look forward).
    - `c_gamma`: [1.7438903314, 1.7579369712], means the distance between camera and moon in OPENGL unit. The real distance is 200m ~ 10,000m.
    - `c_theta`: [0, 2pi], means the theta angle of the camera in the ball coordinate of moon.
    - `c_phi`: [0, pi], means the phi angle of the camera in the ball coordinate of moon.
    - `p_theta`: [0, 2pi], means the theta angle of the reference point about the optical axis in the ball coordinate of moon.
    - `p_phi`: [0, pi], means the phi angle of the reference point about the optical axis in the ball coordinate of moon.
- The goal of this experiment is to predict the position of the camera and its optical axis in the ball coordinate based on the Moon image.

## Enviroment
- Anaconda 3
- python 3.7
- pip install
    - torch==1.3.0
    - tensorboardX==1.9
    - tensorboard==2.0.0
- conda install
    - opencv==3.4.2
    - torchvision==0.4.0

## Config
> Before training or testing you have to adjust config.py.
- Basic Setting
    - Change your logging and CUDA GPU setting.
    - You can turn on parallel gpu setting with `IS_PARALLEL` and `PARALLEL_GPUS`. `PARALLEL_GPUS` means what numbers of GPUs you want to use.
    - `DEVICE`: Choose cpu or gpu.
    - `NET_MODEL`: Which model do you want to use, includes VGG19, Resnet18, Resnet50.
- Path
    - `DATASET_PATH`: Set your own dataset path.
    - `WRITER_PATH`: Set the path of the tensorboard data.
- Dataset
    - `LABEL_TYPE`: Set your labels in this list.
    - `LABEL_NUM`: How many labels do you have.
    - `DATASET_TYPE`: What do you call your dataset.
    - `SPLIT_DATASET_SIZE`: I split my dataset in this way.
    - `SUBDIR_NUM`: In every splited dataset, I also split more directories for data.
    - `DATASET_SIZE`: How many data in each dataset.
    - You have to follow your own dataset to adjust your split way.
- Loss Function
    - `GAMMA_RADIUS`: How long the moon radius is in the OpenGL unit.
    - `GAMMA_UNIT`: Convert 1 OpenGL unit to x km unit.
    - `GAMMA_RANGE`: We define the uperbound of our cammera respective to the moon is 10 km.
    - `CONSTANT_WEIGHT`: How much do you punish the constant of the predict direction.
    - For example, although the distance between 100 * 2pi and 1 * 2pi is zero, we want to predict 1 * 2pi more than 100 * 2pi, so `CONSTANT_WEIGHT` is to add a penalty to the constant of the direction.
    - As a result, 100 * 2pi will take more penalty than 1 * 2pi.
- Visualization
    - `LOG_STEP`: How many steps to record a loss on the tensorboard and print on the console.
    - `TSNE_EPOCH`: How many epochs to record one tsne on the tensorboard.
    - `TSNE_STEP`: How many images to record one tsne in a epoch.
    - `EXPERIMENT_NAME`: You can use your own name of experiment on the tensorboard.
- Hyperparameters
    - All hyperparameters list in this section. You can change any of them by your own way.
    - `EPOCH_NUM`: How many epochs do you want to train.
    - `BATCH_SIZE`: The batch size.
    - `LEARING_RATE`: The learning rate of your optimizer.
    - `MOMENTUM`: Because I use SGD to optimize, I have to set up the momentum of it.

## Data Prepocessing
- Normalize the target range to [0, 1], since the range of gamma is not equal to the range of direction.
- I Load the image in gray scale, since I think the color information is not really important to the position regression.
- All sizes of images are 800 x 600, it make the training time too slow, I use gaussian pyramid down method to resize it to 400 x 300.
- Because the light intensity of images is generally low, Using histogram equalization to strengthen edges and textures of the moon image. See the effect below. Left one is the original image, The other one is equalized image.
<img src="https://github.com/hank-kuo-cs/EVA-Space-Center/blob/master/src/Equalization.png" height="80%" width="80%">

## Network Architecture
- The network model is VGG19 + Global Average Pooling Layer + Fully Connected Network, you can see the simple graph of architecture below. 
- I imported ResNet to the config. You can use it or other models.
<img src="https://github.com/hank-kuo-cs/EVA-Space-Center/blob/master/src/Network_Architecture.png" height="80%" width="80%">

## Custom Loss Function - BCMSE & BCL1
- Because phi & theta are direction of circles, we can't use normal loss function to calculate the distance loss.
- I design custom loss function based on MSE & L1 loss function to handle this issue.
- If the parameter is phi or theta, I will calculate the shortest distance in the circle between predict direction and the target direction. This is an alternation of MSE & L1. They called `Ball Coordinate MSE` (BCMSE) and `Ball Coordinate L1` (BCL1).
- The graph below is an example of calculating BCMSE between a prediction phi and a target phi.
- In the graph, we can see the MSE(phi_predict, phi_target) = (1.5pi)^2, but BCMSE will see the distance between them as 0.5 pi. It can tell the real distance between angles and get the correct MSE loss.
<img src="https://github.com/hank-kuo-cs/EVA-Space-Center/blob/master/src/BCMSE.png" height="50%" width="50%">

## Train
### Command Line Usage
- Just quickly start by the command below.
```bash
python train.py
```
- `train.py` will look for the newest model in the `checkpoint` directory to train. If there is no any model, it will train from scratch.
- There are some arguments you can use:
    - `-m checkpoint/model_epochxxx.pth`: Choose a particular pretrained model to continue training.
    - `-s`: Train from scratch.

## Test
### Command Line Usage
- Just quickly start by the command below.
```bash
python test.py
```
- `test.py` will look for the newest model in the `checkpoint` directory to test.
- There are some arguments you can use:
    - `-m checkpoint/model_epochxxx.pth`: Choose a particular pretrained model to test.
    - `-v`: Test the validation set
    - `-am`: Test all models and record the error percentage and tsne on the tensorboard.

## Tensorboard
- To run the tensorboard, you have to move the directory where your `WRITER_PATH` at there.
- Input the command:
```bash
tensorboard --logdir=<your dir> --bind_all
```
- And you can check the tensorboard on the website, the ip and port will be assigned by tensorboard.

## Visualization
### Train
- The graphs below are loss graph & tsne graph at epoch100.
<img src="https://github.com/hank-kuo-cs/EVA-Space-Center/blob/master/src/SGD_lr_1e-3/train/loss.png" height="80%" width="80%">
<img src="https://github.com/hank-kuo-cs/EVA-Space-Center/blob/master/src/SGD_lr_1e-3/train/tsne/epoch100-gamma.png" height="50%" width="50%">
<img src="https://github.com/hank-kuo-cs/EVA-Space-Center/blob/master/src/SGD_lr_1e-3/train/tsne/epoch100-phi.png" height="50%" width="50%">
<img src="https://github.com/hank-kuo-cs/EVA-Space-Center/blob/master/src/SGD_lr_1e-3/train/tsne/epoch100-theta.png" height="50%" width="50%">

## Result
### Error Percentage
- Epoch 100:
    - Gamma error percentage: 2.53%
    - Phi error percentage: 0.75%
    - theta error percentage: 0.45%
    - total error percentage: 1.24%
- It means the error of the predicted distance between camera and moon is ± 0.380 km
- And the error of the predicted angle between camera and moon, phi is ± 2.63°, and theats is ± 1.62°
## To Do List
- [ ] Change dataset from ball coordinate to xyz coordinate. Because u is in the xyz coordinate and c & p are in the ball coordinate.
- [ ] The range of gamma distance in the OpenGL is too small, it would loss detail information when calculating, because there are too many digits after decimal point.
- [ ] Regress 5 parameters (c_gamma, c_theta, c_phi, p_theta, p_phi).
- [ ] Regress 8 parameters (c_gamma, c_theta, c_phi, p_theta, p_phi, u_x, u_y, u_z).

