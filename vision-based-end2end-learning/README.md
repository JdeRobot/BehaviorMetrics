# Vision-based end-to-end learning

## Info

- More detailed info at my wiki https://jderobot.org/Vmartinezf-tfm

- More detailed info at my Github-pages https://roboticslaburjc.github.io/2017-tfm-vanessa-fernandez/

## Usage

- First of all, we need to install the JdeRobot packages. We need two packages: [JdeRobot-base](https://github.com/JdeRobot/base) and [JdeRobot-assets](https://github.com/JdeRobot/assets). You can follow [this tutorial](https://github.com/JdeRobot/base#getting-environment-ready) for the complete installation.

- Download dataset from [this link]().

- Clone the repository :

    ```bash
    git clone https://github.com/RoboticsLabURJC/2017-tfm-vanessa-fernandez.git
    ```

- Create and activate a virtual environment:

    ```bash
    virtualenv -p python2.7 virtualenv --system-site-packages 
    ```

- Install `requirements`:

  ```bash
  pip install -r requirements.txt
  ```

Launch Gazebo with the f1 world through the command
```bash
roslaunch /opt/jderobot/share/jderobot/gazebo/launch/f1_1_simplecircuit.launch
```

Then you have to execute the application, which will incorporate your code:

```bash
python2 driver.py driver.yml
```

## Solutions

### Train Classification Network

To train the classification network to run the file:

```bash
cd 2017-tfm-vanessa-fernandez/Follow Line/dl-driver/Net/Keras/ 
python classification_train.py
```

When the program is running it will ask for data to know the characteristics of the network to be trained:

1. **Choose one of the options for the number of classes:** *Choose the number of classes you want, typically 4-5 for `v` parameter and 2, 7 or 9 for `w` parameter. Depending on the dataset created, there are different classifications in the json depending on the number of classes for each screen.*

2. **Choose the variable you want to train:** `v` or `w`: *here you put `v` or `w` depending on the type of speed you want to train (traction or turn).*

3. **Choose the type of image you want:** `normal` or `cropped`: *you can choose `normal` or `cropped`. If you want to train with the full image you have to choose `normal` and if you want to train with the cropped image choose `cropped`.*

4. **Choose the type of network you want: normal, biased or balanced:**** *this option refers to the type of dataset or training you want.* 

   *The documentation talks about having on the one hand a training with the whole dataset without any type of treatment of the number of images for each class (there were many more straight lines than curves) or using a balanced dataset that we created keeping the same number of images for each class (v and w),* 

   *To train with that configuration, set the `normal` option. To train with balanced dataset, set the option: `balanced`.*

   *And finally, `biased` refers to you training with the full dataset (unbalanced) but in training you put weights to each class with `class_weigth` type.*

   ```python
   class_weight = {0: 4., 1: 2., 2: 2., 3: 1., 4:2., 5: 2., 6: 3.}
   ```

   *then with this option you can give more weight to the kinds of curves than straight lines. In that example the class 0 is the class `radically_left` and the 6 would be `radically_right`. The option that worked best was that of 'biased'.*

5. **Choose the model you want to use:** `lenet`, `smaller_vgg` or `other`: *Here you have to choose the model you want to train. The option that offers the best results is `smaller_vgg`. The `lenet` model gave very bad results because it was very basic. The `other` model loaded another model that gives worse results. The files containing the network models as such are in the folder `models/`. For classification you have them in `classification_model.py` for regression in `model_nvidia.py`.*
