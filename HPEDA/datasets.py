import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2
import utils

from PIL import Image, ImageFilter


def convertBGR2RGB(imgArray):
    rgbArray = []
    for img in imgArray:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgbArray.append(rgb)
    return np.asarray(rgbArray)


# Data from numpy file
class Pose_Synth_NPY(Dataset):
    def __init__(self, data_dir, filename_path = '', transform = None):

        self.data_dir = data_dir
        self.X_train_data = np.load(os.path.join(data_dir, 'rgbData.npy'))  # [0:50000]
        self.y_train_data = np.load(os.path.join(data_dir, 'poseData.npy'))  # [0:50000]
        self.length = len(self.X_train_data)

    def __getitem__(self, index):

        img = self.X_train_data[index]
        img = img.reshape(64, 64, 3)


        # We get the pose in radians
        pose = self.y_train_data[index]
        # And convert to degrees.
        pitch = pose[1]
        yaw = pose[0]
        roll = pose[2]

        # Data Aug Start

        # img = Image.fromarray(img)
        # # Flip?
        # rnd = np.random.random_sample()
        # if rnd < 0.5:
        #     yaw = -yaw
        #     roll = -roll
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #
        # # Blur?
        # rnd = np.random.random_sample()
        # if rnd < 0.2:
        #     img = img.filter(ImageFilter.BLUR)
        # img = np.asarray(img)

        # Data Aug Ends


        img = torch.from_numpy(img.transpose(-1, 0, 1))
        img = img.float().div(255)


        cont_labels = torch.FloatTensor([yaw, pitch, roll])



        return img, cont_labels

    def __len__(self):
        return self.length


# Data from Raw with Textured Background
class Pose_Synth_Raw(Dataset):
    def __init__(self, data_dir, filename_path, transform = None, img_ext='.jpg'):

        self.data_dir = data_dir
        self.img_ext = img_ext

        filename_list = utils.get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.length = len(filename_list)

    def __getitem__(self, index):

        img = cv2.imread(os.path.join(self.data_dir, self.X_train[index].split(',')[0]),
                         cv2.IMREAD_UNCHANGED)
        try:
            img = cv2.resize(img, (64, 64))



            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.reshape(64, 64, 3)

            img = torch.from_numpy(img.transpose(-1, 0, 1))
            img = img.float().div(255)

            txt_path = os.path.join(self.data_dir, self.X_train[index].split(',')[1])

        except:
            test = 55
            print('error')




        # We get the pose in degrees

        pose = utils.get_ypr_from_txt(txt_path)
        yaw = pose[0]
        pitch = pose[1]
        roll = pose[2]

        # if pose[2]>5:
        #     roll = pose[2] - 1
        # elif pose[2]<-5:
        #     roll = pose[2] +1
        # else:
        #     roll = pose[2]

        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        # if self.transform is not None:
        #     img = self.transform(img)

        return img, cont_labels

    def __len__(self):
        return self.length


# Data from Raw with Real background
class Pose_Synth_Raw_RB(Dataset):
    def __init__(self, data_dir, filename_path, transform = None, img_ext='.jpg'):

        self.data_dir = data_dir
        self.img_ext = img_ext

        filename_list = utils.get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.length = len(filename_list)

    def __getitem__(self, index):

        idx = self.X_train[index].split('/')[-1]
        rgbPath = os.path.join(self.data_dir, self.X_train[index].split('/')[0],
                               self.X_train[index].split('/')[1], 'rgb' + idx + '.jpg')
        posePath = os.path.join(self.data_dir, self.X_train[index].split('/')[0],
                                self.X_train[index].split('/')[1], 'pose' + idx + '.txt')

        img = cv2.imread(rgbPath, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.reshape(64, 64, 3)
        img = torch.from_numpy(img.transpose(-1, 0, 1))
        img = img.float().div(255)

        # We get the pose in degrees
        # txt_path = os.path.join(self.data_dir, self.y_train[index])
        pose = utils.get_ypr_from_txt(posePath)
        yaw = pose[0]
        pitch = pose[1]
        roll = pose[2]

        # # Flip?
        # rnd = np.random.random_sample()
        # if rnd < 0.5:
        #     yaw = -yaw
        #     roll = -roll
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #
        # # Blur?
        # rnd = np.random.random_sample()
        # if rnd < 0.05:
        #     img = cv2.flip(ImageFilter.BLUR)

        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        # if self.transform is not None:
        #     img = self.transform(img)

        return img, cont_labels

    def __len__(self):
        return self.length


# Data from Raw BIWI
class Pose_BIWI_Raw(Dataset):
    def __init__(self, data_dir, filename_path, transform = None, img_ext='.jpg'):

        self.data_dir = data_dir
        self.img_ext = img_ext

        filename_list = utils.get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.length = len(filename_list)

    def __getitem__(self, index):

        img = cv2.imread(os.path.join(self.data_dir, self.X_train[index].split(',')[0]),
                         cv2.IMREAD_UNCHANGED)



        img = cv2.resize(img, (64, 64))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape(64, 64, 3)
        img = torch.from_numpy(img.transpose(-1, 0, 1))
        img = img.float().div(255)

        # We get the pose in degrees
        txt_path = os.path.join(self.data_dir, self.X_train[index].split(',')[1])
        pose = utils.get_ypr_from_txt(txt_path)
        yaw = pose[0]
        pitch = pose[1]
        roll = pose[2]

        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        # if self.transform is not None:
        #     img = self.transform(img)

        return img, cont_labels

    def __len__(self):
        return self.length


# Data from BIWI Numpy
class Pose_BIWI_NPY(Dataset):
    def __init__(self, data_dir, filename_path = '', transform = None):

        self.data_dir = data_dir
        rgbData = np.load(os.path.join(data_dir, 'testImg.npy'))  # 'trnImg.npy'
        self.X_train_data = convertBGR2RGB(rgbData)
        self.y_train_data = np.load(os.path.join(data_dir, 'testPose.npy'))  # 'trnPose.npy'
        self.length = len(self.X_train_data)

    def __getitem__(self, index):

        img = self.X_train_data[index]
        img = img.reshape(64, 64, 3)
        img = torch.from_numpy(img.transpose(-1, 0, 1))
        img = img.float().div(255)

        # We get the pose in radians
        pose = self.y_train_data[index]
        # And convert to degrees.
        pitch = pose[1]
        yaw = pose[0]
        roll = pose[2]

        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        # if self.transform is not None:
        #     img = self.transform(img)

        return img, cont_labels

    def __len__(self):
        return self.length


# Data from numpy file
class Pose_Synth_NPYDA(Dataset):
    def __init__(self, data_dir, filename_path = '', transform = None):

        # self.data_dir = data_dir
        self.X_train_data = np.load('models/biwiSynImg.npy')
        self.y_train_data = np.load('models/biwiSynPose.npy')
        biwi_data = np.load('../../Data/RealNPZ/BIWI/testImg.npy')
        self.biwi_train_data = convertBGR2RGB(biwi_data)
        self.length = len(self.X_train_data)

    def __getitem__(self, index):

        img = self.X_train_data[index]
        img = img.reshape(64, 64, 3)

        # We get the pose in radians
        pose = self.y_train_data[index]
        # And convert to degrees.
        pitch = pose[1]
        yaw = pose[0]
        roll = pose[2]

        img = torch.from_numpy(img.transpose(-1, 0, 1))
        img = img.float().div(255)

        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        biwiImg = self.biwi_train_data[index]
        biwiImg = biwiImg.reshape(64, 64, 3)
        biwiImg = torch.from_numpy(biwiImg.transpose(-1, 0, 1))
        biwiImg = biwiImg.float().div(255)

        return img, cont_labels, biwiImg

    def __len__(self):
        return self.length