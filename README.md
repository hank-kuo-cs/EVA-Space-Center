# EVA-Space-Center
## Introduction
##### Time
    - A project started from 2019 June. 
##### Goal
    - It's aims to predict the position and the pose of an aircraft from a single Moon image.
- Data consists of 100,000 moon images from random angles and distance.
- 80,000 train data；10,000 test data；10,000 valid data.
- The label(target/ground truth): `c_gamma`, `c_theta`, `c_phi`, `p_gamma`, `p_theta`, `p_phi`, `u_x`, `u_y`, `u_z`
- The gobal world coordinate is based on Moon center, all vectors are based on the origin of the global world coordinate. 
- We consider the spherical coordinate in physics view.   <img src="https://github.com/charleschiu2012/EVA-Space-Center-Data-Generate/blob/master/src/360px-3D_Spherical.svg.png"  height="20%" width="20%">
- The meaning of this 9 parameters list below:
    - `c_gamma`: gamma of the camera position.
    - `c_theta`: theta of the camera position.
    - `c_phi`: phi of the camera position.
    - `p_gamma`: gamma of the optical axis' end point. However, the setting of p_gamma doesn't cause any difference of where cemera look at nor the image.
    - `p_theta`: theta of the optical axis' end point.
    - `p_phi`: phi of the optical axis' end pount.
    - `u_x`: x componet of camera's normal vecter.
    - `u_y`: y componet of camera's normal vecter.
    - `u_z`: z componet of camera's normal vecter.
- The range of this 9 parameters list below:
    - `c_gamma`: [1.74308766628, 1.75292031414] in OpenGL unit --> [1737.3,  1747.1] km, 200m ~ 10,000m above Moon surface.
    - `c_theta`: [0, 2pi] radian
    - `c_phi`: [0, pi] radian
    - `p_gamma`: [0, 1.742887] in OpenGL unit --> [0, 1737.1] km, radius of the Moon.
    - `p_theta`: [0, 2pi] radian
    - `p_phi`: [0, pi] radian
    - `u_x`: [-1, 1] no unit, since the normal vector is normalized.
    - `u_y`: [-1, 1] no unit , since the normal vector is normalized.
    - `u_z`: [-1, 1] no unit, since the normal vector is normalized.
```c++
    void gluLookAt(	GLdouble eyeX, GLdouble eyeY, GLdouble eyeZ,
                        GLdouble centerX, GLdouble centerY, GLdouble centerZ,
                        GLdouble upX, GLdouble upY, GLdouble upZ
                   );
```
- gluLookAt Parameters Meaning:
    - eyeX, eyeY, eyeZ
        - Specifies the position of the eye point.
    - centerX, centerY, centerZ
        - Specifies the position of the reference point.
    - upX, upY, upZ
        - Specifies the direction of the up vector.
- See more definition of gluLookAt at https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml

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
    - You can turn on parallel gpu setting with `IS_PARALLEL` and `PARALLEL_GPUS`.
    - When you set `IS_PARALLEL` to True, you have to delete the code `os.environ['CUDA_VISIBLE_DEVICES']`.
    - `NET_MODEL` means which model do you want to use, includes VGG19, Resnet18, Resnet50.
- Path
    - `DATASET_PATH`: You must set your own dataset path.
    - `WRITER_PATH`: Set the path of the tensorboard.
- Dataset
    - You can adjust your own split way on the train, test, validation set.
    - But advise you not change this setting, maybe a bug will happen when loading data.
- Units
    - Units about OpenGL and the Moon. 
- Constraints
    - Only you can change is `CONSTANT_WEIGHT`. It means how much do we punish the constant of the predict direction. But advise not to change it.
    - For example, although the distance between 100 * 2pi and 1 * 2pi is zero, we want to predict 1 * 2pi more than 100 * 2pi, so `CONSTANT_WEIGHT` is to add a penalty to the constant of the direction.
    - As a result, 100 * 2pi will take more penalty than 1 * 2pi.
    - `LIMIT` is about the range of each parameters.
- Visualization
    - `LOG_STEP`: How many steps to record a loss on the tensorboard and print on the console.
    - `TSNE_EPOCH`: How many epochs to record one tsne on the tensorboard.
    - `TSNE_STEP`: How many images to record one tsne in a epoch.
    - `EXPERIMENT_NAME`: You can use your own name of experiment on the tensorboard.
- Hyperparameters
    - All hyperparameters list in this section. You can change any of them by your own way.

## Data Prepocessing
- Normalize the target range to [0, 1], since the range of gamma is not equal to the range of direction.
- I Load the image in gray scale, since I think the color information is not really important to the position regression.
- All sizes of images are 800 x 600, it make the training time too slow, I use gaussian pyramid down method to resize it to 400 x 300.
- Because the light intensity of images is generally low, Using histogram equalization to strengthen edges and textures of the moon image. See the effect below. Left one is the original image, The other one is equalized image.
<img src="https://github.com/hank-kuo-cs/EVA-Space-Center/blob/master/src/Equalization.png" height="80%" width="80%">

## Network Architecture
- The network model is ResNet18 + Fully Connected Network, you can see graph of architecture below. 
- Of course, you can choose your own model, but be aware that input image is of one channel, your model should handle that problem.
<img src="https://github.com/hank-kuo-cs/EVA-Space-Center/blob/charles/src/VGG19.png" height="80%" width="80%">

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
    - `-m model_epochxxx.pth`: Choose a particular pretrained model to continue training.
    - `-s`: Train from scratch.

## Test
### Command Line Usage
- Just quickly start by the command below.
```bash
python test.py
```
- `test.py` will look for the newest model in the `checkpoint` directory to test.
- There are some arguments you can use:
    - `-m model_epochxxx.pth`: Choose a particular pretrained model to test.
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
- [ ] Add check NaN function
- [ ] Check vector if it is in positive domain 


