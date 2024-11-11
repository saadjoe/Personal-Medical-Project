import os
import shutil

IMG_PATH = "brain_tumor_dataset/"

# Split the data into training, validation, and testing sets
for CLASS in os.listdir(IMG_PATH):
    if not CLASS.startswith("."):
        IMG_NUM = len(os.listdir(IMG_PATH + CLASS))
        for n, FILE_NAME in enumerate(os.listdir(IMG_PATH + CLASS)):
            img = IMG_PATH + CLASS + "/" + FILE_NAME
            if n < 5:
                shutil.copy(img, "data/TEST/" + CLASS.upper() + "/" + FILE_NAME)
            elif n < 0.8 * IMG_NUM:
                shutil.copy(img, "data/TRAIN/" + CLASS.upper() + "/" + FILE_NAME)
            else:
                shutil.copy(img, "data/VAL/" + CLASS.upper() + "/" + FILE_NAME)
