## PR-Strawberry Project:strawberry:
### Project Requirements
This one is related to smart farm image analysis, and you will be provided with [Strawberry data](https://pan.baidu.com/s/1GujktkCBsXsgk6r9tHkXVA?pwd=6u85):

**Image Files** altogether we have 502 images. All the images were labeled according to the ripening stage, *unripe, part-ripe, and ripe*.

**CSV File** Annotations.csv consists of *file_name, Ripe label, Brix (for taste), and Acidity values*.

You are required to develop PR models that can do the following analysis for any new image of strawberry:
1. correctly classify (i.e. “ripe” “part-ripe”, and “unripe”) for any new image of strawberry.
2. estimate the Brix value for any new image of strawberry.
3. estimate the Acidity value for any new image of strawberry.

### Solve the Problem 1
**Correctly classify (i.e. “ripe” “part-ripe”, and “unripe”) for any new image of strawberry.**
1. Resize the images
From the *Image Files*, 502 images have various size. In order to put these images into our later *CNN Network*, images should be resize to the same size, for an instance, 96x96 pixels(images can be set to the size that everyone wants. Considering the better size for the *CNN Network*, 96x96 pixels size is committed).
It should be noted that the original data set images are 4-channel <u>RGBA images</u> and need to be converted into 3-channel <u>RGB images</u> so that they can be fed into the CNN model training.

    Details can be seen in [1_picture_resize.py]()


2. Clean the noises in *Annotations.csv*
Actually, this problem is not founded until images and *Annotations.csv* file put into *CNN Network*. Python told that there were some images that could not be found in *Image Files*. Thus, noises in *Annotations.csv* is founded.
Images' filenames and labels don't correspond to the *Annotations.csv*, about 22 images have noises.
Obviously, compared with strings, it is more convenient to directly process integer numbers, so we replace unripe, partripe, and ripe of label in the csv file according to the following rules:

    >unripe=0
    >partripe=1
    >ripe=2

    Details can be seen in [2_clean_Annotations.py]()

3. Classification:3 classes
Some settings:
    >Model: VGG16
    >Learning_rate = 1e-4
    >optimizer: Adam
    >Loss function: nn.CrossEntropyLoss()

    After starting training, you can use *tensorboard* to view the training process, use the Vscode plug-in or enter the following code in the terminal:
    ```
    tensorboard --logdir="logs
    tensorboard --logdir=../logs --port=8006  # Alternative method when interface is occupied
    ```

    Details can be seen in [3_classification.py]()

4. Verification
finally, maybe this classify-model might be used in the world:innocent:.
Details can be seen in [4_verification.py]()

### Write behind
In fact, I also made a simple tkinter visual interface, but it is very simple. I will upload it when I have the opportunity in the future.

Maybe in the future, I will solve the Problem 2 and 3:disappointed_relieved:.
