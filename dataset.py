import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import cv2

class brain_ImageData(Dataset):

    # Class constuctor
    def __init__(self):

        #Variables that will hold the training data and the testing data
        self.img_Train  = np.empty((0, 128, 128, 3), dtype=np.float32)
        self.label_Train = np.empty((0,), dtype=np.float32)
        self.img_Test   = np.empty((0, 128, 128, 3), dtype=np.float32)
        self.label_Test  = np.empty((0,), dtype=np.float32)

        #Variable that determines if we are in Train mode or Test Mode
        self.mode  ='train'

        tumor = []     # Array with all yes images
        healthy = []   # Array with al healthy images

        Yes_path = './yes/*.jpg'
        No_Path  = './no/*.jpg'

        # For loop appends all yes images to tumor array, resizes them, and sets configures values to r,g,b
        # cv2 - reads in BGR by default so configured data into RGB
        for f in glob.iglob(Yes_path):
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img,(128,128))
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            tumor.append(img)

        # For loop appends all yes images to tumor array, resizes them, and sets configures values to r,g,b
        for f in glob.iglob(No_Path):
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img,(128,128))
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            healthy.append(img)

        # Creates numpy arrays for the data images
        healthy = np.array(healthy, dtype=np.float32)
        tumor   = np.array(tumor,   dtype=np.float32)

        #Creates Label arrays for the data images
        tumor_Label   = np.ones(tumor.shape[0],   dtype=np.float32)
        healthy_Label = np.zeros(healthy.shape[0], dtype=np.float32)

        #Concatenate or combine all data into an array of images and an array of Labels
        self.images = np.concatenate((tumor, healthy), axis=0)
        self.labels = np.concatenate((tumor_Label, healthy_Label), axis=0)

    def data_split(self):
        if self.images.size == 0:
            raise RuntimeError("No images found. Make sure ./yes/*.jpg and ./no/*.jpg exist.")
        self.img_Train, self.img_Test, self.label_Train, self.label_Test = train_test_split(
            self.images, self.labels, test_size=0.20, random_state=42, stratify=self.labels
        )  # Divides the data into a train data set and a test data set

    #Returns length or the number of total images
    def __len__(self):
        #Check to see if it returns train data or test data
        if self.mode == 'train':
            # help Pylance: these cannot be None after data_split()
            assert isinstance(self.img_Train, np.ndarray)
            return self.img_Train.shape[0]
        elif self.mode == 'test':
            assert isinstance(self.img_Test, np.ndarray)
            return self.img_Test.shape[0]
        else:
            return 0

    #Returns dictionary at specifed index including the image and the label
    def __getitem__(self, index):
        #Check to see if it returns train data or test data
        if self.mode == 'train':
            assert isinstance(self.img_Train, np.ndarray) and isinstance(self.label_Train, np.ndarray)
            sample = { 'images': self.img_Train[index], 'labels': self.label_Train[index] }
        elif self.mode =='test':
            assert isinstance(self.img_Test, np.ndarray) and isinstance(self.label_Test, np.ndarray)
            sample = { 'images': self.img_Test[index], 'labels': self.label_Test[index] }
        else:
            raise ValueError(f"Unknown mode {self.mode!r}")
        return sample

    #Normalizes data to make all values between 0 and 1 instead of 0 to 255
    def normalize(self):
        self.images = self.images / 255.0
