import numpy as np
import torch
import torch.nn as nn
import cv2

from HPEDA.FSANetDA import FSANet
import datasets


# import matplotlib.pyplot as plt


def predict(model, rgb, pose, batch_size=4, verbose=False):
    N = len(rgb)

    bs = batch_size

    predictions = []
    testSetPose = []

    for i in range(N // bs):
        x = rgb[(i) * bs:(i + 1) * bs, :, :, :]

        # Compute results
        true_y = pose[(i) * bs:(i + 1) * bs, :]

        x1 = x.transpose(0, -1, 1, 2)
        x2 = torch.from_numpy(x1).float().div(255)
        x3, _ = model(x2.cuda(), alpha=0.1)
        pred_y = x3.detach().cpu().numpy()

        # print(true_y.shape, pred_y.shape)

        predictions.append(pred_y)
        testSetPose.append(true_y)

    p_data = np.concatenate(predictions, axis=0)
    y_data = np.concatenate(testSetPose, axis=0)

    print(p_data.shape, y_data.shape)
    return p_data, y_data


def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"]


def convertBGR2RGB(imgArray):
    rgbArray = []
    for img in imgArray:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgbArray.append(rgb)
    return np.asarray(rgbArray)


_IMAGE_SIZE = 64
stage_num = [3, 3, 3]
lambda_d = 1
num_classes = 3

num_capsule = 3
dim_capsule = 16
routings = 2

num_primcaps = 7 * 3
m_dim = 5
S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

_TEST_DB_AFLW = "AFLW2000"
_TEST_DB_BIWI1 = "BIWI_noTrack"
_TEST_DB_BIWI2 = "BIWI_Test"
_TEST_DB_POINTING = "Pointing"
_TEST_DB_SASE = "SASE"
_TEST_DB_SYNTHETIC = "SYNTHETIC"

# test_db_list = [_TEST_DB_AFLW, _TEST_DB_BIWI1, _TEST_DB_POINTING, _TEST_DB_SASE]  # _TEST_DB_AFLW, , _TEST_DB_BIWI2

test_db_list = [_TEST_DB_SYNTHETIC]  # [_TEST_DB_BIWI1]

# get device GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---UNet with modified loss (surface normal) ------#
model = FSANet(S_set).cuda()

# get multiple GPU
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)

# ----- Get the optimum epoch ----- #
if True:

    epT = 0
    MAET = 10.0

    for ep in range(0, 40):
        modelPath = 'models/BIWIRaw_11-25-2020_21-14-54-n1484-e40-bs8-lr0.0001/weights.epoch{0}_model.pth'. \
            format(str(ep))

        # print(modelPath)

        model.load_state_dict(torch.load(modelPath))
        model.eval()

        if True:

            for test_db_name in test_db_list:

                if test_db_name == _TEST_DB_AFLW:
                    image, pose = load_data_npz('../Data/RealNPZ/AFLW2000.npz')

                elif test_db_name == _TEST_DB_BIWI1:
                    image = np.load('/mnt/fastssd/Shubhajit_Stuff/HPECode/Data/BIWI/testImg.npy')
                    pose = np.load('/mnt/fastssd/Shubhajit_Stuff/HPECode/Data/BIWI/testPose.npy')
                    # image, pose = load_data_npz('../Data/RealNPZ/BIWI_noTrack.npz')

                elif test_db_name == _TEST_DB_BIWI2:
                    image, pose = load_data_npz('../Data/RealNPZ/BIWI_test.npz')

                elif test_db_name == _TEST_DB_SASE:
                    image = np.load('../Data/RealNPZ/SASERgbData.npy')
                    pose = np.load('../Data/RealNPZ/SASEPoseData.npy')

                elif test_db_name == _TEST_DB_POINTING:
                    image = np.load('../Data/RealNPZ/PointingRgbData.npy')
                    pose = np.load('../Data/RealNPZ/PointingPoseData.npy')

                elif test_db_name == _TEST_DB_SYNTHETIC:
                    image = np.load('/mnt/fastssd/Shubhajit_Stuff/HPECode/Data/SynData/rgbData.npy')[0:12000]
                    pose = np.load('/mnt/fastssd/Shubhajit_Stuff/HPECode/Data/SynData/poseData.npy')[0:12000]

                # image = convertBGR2RGB(image)

                x_data = []
                y_data = []

                for i in range(0, pose.shape[0]):
                    temp_pose = pose[i, :]
                    # if (np.max(temp_pose[0]) <= 60.0 and np.min(temp_pose[0]) >= -60.0) and \
                    #         (np.max(temp_pose[1]) <= 50.0 and np.min(temp_pose[1]) >= -50.0) and \
                    #         (np.max(temp_pose[2]) <= 40.0 and np.min(temp_pose[2]) >= -40.0):
                    if np.max(temp_pose) <= 90.0 and np.min(temp_pose) >= -90.0:
                        x_data.append(image[i, :, :, :])
                        y_data.append(pose[i, :])
                x_data = np.array(x_data)
                y_data = np.array(y_data)

                p_data, y_data = predict(model, x_data, y_data, batch_size=64)

                # p_data[:, 2] = -p_data[:, 2]

                pose_matrix = np.mean(np.abs(p_data - y_data), axis=0)
                MAE = np.mean(pose_matrix)
                yaw = pose_matrix[0]
                pitch = pose_matrix[1]
                roll = pose_matrix[2]
                print('\n--------------------------------------------------------------------------------')
                print(test_db_name, ep,
                      '   :   MAE = %3.3f, [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]' % (MAE, yaw, pitch, roll))
                print('--------------------------------------------------------------------------------')

                if MAE < MAET:
                    epT = ep
                    MAET = MAE

                    result = (MAE, yaw, pitch, roll)

    print('BIWI: ', epT, '   :   MAE = %3.3f, [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]' % result)

    # 59

# ----- Result for optimum epoch ----- #
if False:

    modelPath = r'../models/MySynth_09-05-2020_18-50-28-n8858-e90-bs8-lr0.0001/weights.epoch72_model.pth'

    # modelPath = r'models/MySynthTrans_10-13-2020_03-10-10-n168-e50-bs8-lr0.0001/weights.epoch39_model.pth'

    # print(modelPath)

    model.load_state_dict(torch.load(modelPath))
    model.eval()

    if True:

        for test_db_name in test_db_list:

            if test_db_name == _TEST_DB_AFLW:
                image, pose = load_data_npz('../Data/RealNPZ/AFLW2000.npz')

            elif test_db_name == _TEST_DB_BIWI1:
                image = np.load('../../Data/RealNPZ/BIWI/testImg.npy')
                pose = np.load('../../Data/RealNPZ/BIWI/testPose.npy')

                # image, pose = load_data_npz('../Data/RealNPZ/BIWI_noTrack.npz')

            elif test_db_name == _TEST_DB_BIWI2:
                image, pose = load_data_npz('../Data/RealNPZ/BIWI_test.npz')

            elif test_db_name == _TEST_DB_SASE:
                image = np.load('../Data/RealNPZ/SASERgbData.npy')
                pose = np.load('../Data/RealNPZ/SASEPoseData.npy')

            # image = np.delete(image,[1394, 1398, 1403],axis=0)
            # pose = np.delete(pose,[1394, 1398, 1403],axis=0)

            image = convertBGR2RGB(image)

            x_data = []
            y_data = []

            for i in range(0, pose.shape[0]):
                temp_pose = pose[i, :]
                # if ((np.max(temp_pose[0]) <= 90.0 and np.min(temp_pose[0]) >= -90.0) and \
                #         (np.max(temp_pose[1]) <= 10.0 and np.min(temp_pose[1]) >= -10.0) and \
                #         (np.max(temp_pose[2]) <= 10.0 and np.min(temp_pose[2]) >= -10.0)):
                if np.max(temp_pose) <= 90.0 and np.min(temp_pose) >= -90.0:
                    x_data.append(image[i, :, :, :])
                    y_data.append(pose[i, :])
            x_data = np.array(x_data)
            y_data = np.array(y_data)

            p_data, y_data = predict(model, x_data, y_data, batch_size=8)

            # l = 1
            #
            # for y, p in zip(y_data, p_data):
            #     print(l, y, p)
            #     l = l+1

            pose_matrix = np.mean(np.abs(p_data - y_data), axis=0)
            MAE = np.mean(pose_matrix)
            yaw = pose_matrix[0]
            pitch = pose_matrix[1]
            roll = pose_matrix[2]
            print('\n--------------------------------------------------------------------------------')
            print(test_db_name,
                  '   :   MAE = %3.3f, [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]' % (MAE, yaw, pitch, roll))
            print('--------------------------------------------------------------------------------')

if False:
    modelPath = r'models/MySynth_09-05-2020_18-50-28-n8858-e90-bs8-lr0.0001/weights.epoch72_model.pth'

    data_dir = '/mnt/fastssd/Shubhajit_stuff/DA-Code/HeadPoseCode/Data/BIWI/FRData/'
    filename_path = '/mnt/fastssd/Shubhajit_stuff/DA-Code/HeadPoseCode/Data/BIWI/FRData/data.txt'

    batch_size = 8

    model.load_state_dict(torch.load(modelPath))
    model.eval()
    pose_dataset = datasets.Pose_BIWI_Raw(data_dir, filename_path=filename_path)

    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)

    predictions = []
    testSetPose = []

    for i, (images, cont_labels) in enumerate(test_loader):
        # images = Variable(images).cuda()
        # label_angles = Variable(cont_labels[:, :3]).cuda(non_blocking=True)

        images = images.cuda()
        label_angles = cont_labels[:, :3].cuda()

        # Predict
        angles = model(images)

        pred_y = angles.detach().cpu().numpy()
        true_y = cont_labels.detach().cpu().numpy()

        predictions.append(pred_y)
        testSetPose.append(true_y)

    p_data = np.concatenate(predictions, axis=0)
    y_data = np.concatenate(testSetPose, axis=0)

    pose_matrix = np.mean(np.abs(p_data - y_data), axis=0)
    MAE = np.mean(pose_matrix)
    yaw = pose_matrix[0]
    pitch = pose_matrix[1]
    roll = pose_matrix[2]
    print('\n--------------------------------------------------------------------------------')
    print('BIWI',
          '   :   MAE = %3.3f, [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]' % (MAE, yaw, pitch, roll))
    print('--------------------------------------------------------------------------------')

if False:

    image = np.load('../../Data/RealNPZ/BIWI/testImg.npy')
    pose = np.load('../../Data/RealNPZ/BIWI/testPose.npy')

    x_data = []
    y_data = []

    for i in range(0, pose.shape[0]):
        temp_pose = pose[i, :]
        # if (np.max(temp_pose[0]) <= 60.0 and np.min(temp_pose[0]) >= -60.0) and \
        #         (np.max(temp_pose[1]) <= 50.0 and np.min(temp_pose[1]) >= -50.0) and \
        #         (np.max(temp_pose[2]) <= 40.0 and np.min(temp_pose[2]) >= -40.0):
        if np.max(temp_pose) <= 90.0 and np.min(temp_pose) >= -90.0:
            x_data.append(image[i, :, :, :])
            y_data.append(pose[i, :])
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    for n, img in enumerate(x_data):
        cv2.imwrite('models/biwiTest/{0}.jpg'.format(n), img)

    print('tst')

if False:
    synPose = np.load('/mnt/fastssd/Shubhajit_stuff/DA-Code/HeadPoseCode/Data/SyntheticNpy/poseData.npy')
    synImg = np.load('/mnt/fastssd/Shubhajit_stuff/DA-Code/HeadPoseCode/Data/SyntheticNpy/rgbData.npy')

    synPose = synPose[50000:-1]
    synImg = synImg[50000:-1]

    p_data = np.load('models/pred.npy')

    newPose = []
    newImg = []
    indexL = []

    for p in p_data:
        p = [x + np.random.uniform(-2, 2) for x in p]

        t1 = np.array([p, ] * synPose.shape[0])

        diff = np.mean(np.abs(synPose - t1), axis=1)
        index = np.argmin(diff)

        newPose.append(synPose[index])
        newImg.append(synImg[index])
        indexL.append(index)

    newPoseNPY = np.asarray(newPose)
    newImgNPY = np.asarray(newImg)
    indexNPY = np.asarray(indexL)

    np.save('models/biwiSynPose.npy', newPoseNPY)
    np.save('models/biwiSynImg.npy', newImgNPY)
    np.savetxt('models/biwiSynPoseIndex.txt', indexNPY, fmt='%.0f')

if False:
    # x_data = np.load('models/biwiSynImg.npy')
    # for n, img in enumerate(x_data):
    #     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite('models/biwiSyn/{0}.jpg'.format(n), rgb)

    # y_data = np.load('models/biwiSynPose.npy')
    # np.savetxt('models/biwiSynPose.txt', y_data, fmt='%.3f')

    # synImg = np.load('/mnt/fastssd/Shubhajit_stuff/DA-Code/HeadPoseCode/MyHPECode/HPEDA/models/biwiSynImg.npy')
    # synImg = synImg[50000:-1]

    # for n, img in enumerate(synImg):
    #     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite('models/biwiSyn/{0}.jpg'.format(n), rgb)

    p_data = np.load('models/pred.npy')
    print(p_data.shape)

    biwiimage = np.load('../../Data/RealNPZ/BIWI/testImg.npy')
    biwipose = np.load('../../Data/RealNPZ/BIWI/testPose.npy')
    biwiimage = biwiimage[0:11872]
    biwipose = biwipose[0:11872]

    synimage = np.load('models/biwiSynImg.npy')
    synpose = np.load('models/biwiSynPose.npy')

    print(biwiimage.shape)
    print(synimage.shape)
    print(synpose.shape)


