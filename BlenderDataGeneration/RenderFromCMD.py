import os
import subprocess
import sys
from datetime import datetime

"""
    This code do the following - 
        1. Import the fbx file into a existing blender model
        2. Import missing texture files in blender
        3. Setup camera and other stuff in 3D scene in blender
        4. Setup compositor nodes in Blender
        5. Render from command line 
"""

blenderPath = '/mnt/fastssd/Shubhajit_Stuff/blender-2.81-linux-glibc217-x86_64/blender '

target_folder = r'/mnt/fastssd/Shubhajit_Stuff/dataCreation/male/'
renderScript = r'\captureWithGaze_Texture.py'  #
importfbxScript = r'\importFbx.py'
simpleScnScript = r'\sceneSetup_V1.py'
compositorScript = r'\compositorSetup.py'
importMissingFileScript = r'\ImportMisFileBlender.py'

renderReal = False
# use an existing blend file to add the models and render
# simpleScene.blend    |     headPoseScene.blend
if renderReal:
    blenderScenePath = r'/mnt/fastssd/Shubhajit_Stuff/Environments/SimpleScene/headPoseScene.blend'
else:
    blenderScenePath = r'/mnt/fastssd/Shubhajit_Stuff/Environments/SimpleScene/simpleScene.blend'

sceneName = '_simple'

# ---- Male List ---- #  2,
# idList = [
#     1, 4, 7, 9, 10, 11, 13, 15, 18, 20, 24, 25,
#     28, 31, 35, 37, 38, 40, 41, 42, 43, 44, 45, 46,
#     47, 48, 49, 50, 51, 52, 54, 56, 58, 60, 61, 62,
#     63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
#     75, 76, 77, 78, 79, 80, 81, 82, 83, 84]

# idList = [1, 4, 7, 9, 10, 11, 13, 15, 18, 20, 24, 25]
idList = [1]
flag = False
renderFlag = True

# ----------- Blender FBX setup ---------------- #

if flag:
    for root, dirs, files in os.walk(target_folder):
        for dir in dirs:
            for file in os.listdir(str(root) + '/' + str(dir)):
                if file.endswith(".fbx") & ('Simple' in os.path.join(root, dir, file)):  # & file.startswith('male')
                    # print(os.path.join(root, dir, file).split('/')[-4])
                    id = int(float(os.path.join(root, dir, file).split('/')[-4]))
                    if id in idList:
                        # if (id > 0) & (id < 101):
                        fbxFilePath = os.path.join(root, dir, file)
                        if 'Simple/Neutral' in fbxFilePath:
                            # if True:

                            blenderFile = sceneName.join(fbxFilePath.split('/')[5:7]) + '.blend'
                            blenderFilePath = os.path.join(root, dir, blenderFile)
                            cmd = blenderPath + blenderScenePath \
                                  + " --background -P " \
                                  + importfbxScript + " " + fbxFilePath + " " + blenderFilePath
                            # print(cmd)
                            p = subprocess.call(cmd, shell=True)

# ----------- Blender Import Missing File ---------------- #

if flag:
    for root, dirs, files in os.walk(target_folder):
        for dir in dirs:
            for file in os.listdir(str(root) + '/' + str(dir)):
                if file.endswith(".blend") & file.startswith('male_simple'):
                    # print(os.path.join(root, dir, file))
                    if int(file.split('.')[0][-3:]) in idList:
                        # if (int(file.split('.')[0][-3:]) > 0) & (int(file.split('.')[0][-3:]) < 101):
                        blenderFilePath = os.path.join(root, dir, file)
                        if 'Neutral' in blenderFilePath:
                            pwd = 'galwaydnn'
                            # print(blenderFilePath)
                            cmd = blenderPath + " --background " \
                                  + blenderFilePath + " -P " + importMissingFileScript
                            # print(cmd)
                            p = subprocess.call('echo {} | sudo -S {}'.format(pwd, cmd), shell=True)

# ----------- Blender Model setup ---------------- #

if flag:
    for root, dirs, files in os.walk(target_folder):
        for dir in dirs:
            for file in os.listdir(str(root) + '/' + str(dir)):
                if file.endswith(".blend") & file.startswith('male_simple'):
                    # print(os.path.join(root, dir, file))
                    if int(file.split('.')[0][-3:]) in idList:
                        # if (int(file.split('.')[0][-3:]) > 0) & (int(file.split('.')[0][-3:]) < 101):
                        blenderFilePath = os.path.join(root, dir, file)
                        if 'Neutral' in blenderFilePath:
                            pwd = 'galwaydnn'
                            cmd = blenderPath + " --background " \
                                  + blenderFilePath + " -P " + simpleScnScript
                            # print(cmd)
                            p = subprocess.call('echo {} | sudo -S {}'.format(pwd, cmd), shell=True)

# ----------- Blender Compositor setup ---------------- #

if flag:
    if not renderReal:
        print('compositor called')
        for root, dirs, files in os.walk(target_folder):
            for dir in dirs:
                for file in os.listdir(str(root) + '/' + str(dir)):
                    if file.endswith(".blend") & file.startswith('male_simple'):
                        # print(os.path.join(root, dir, file))
                        if int(file.split('.')[0][-3:]) in idList:
                            # if (int(file.split('.')[0][-3:]) > 0) & (int(file.split('.')[0][-3:]) < 101):
                            blenderFilePath = os.path.join(root, dir, file)
                            if 'Neutral' in blenderFilePath:
                                pwd = 'galwaydnn'
                                # print(blenderFilePath)
                                cmd = blenderPath + " --background " \
                                      + blenderFilePath + " -P " + compositorScript  # importMissingFileScript
                                # print(cmd)
                                p = subprocess.call('echo {} | sudo -S {}'.format(pwd, cmd), shell=True)

# -------- Blender Rendering -------#

if renderFlag:

    c = 0
    for root, dirs, files in os.walk(target_folder):
        for dir in dirs:
            for file in os.listdir(str(root) + '/' + str(dir)):
                if file.endswith(".blend") & file.startswith('male_simple'):
                    if int(file.split('.')[0][-3:]) in idList:
                        # if (int(file.split('.')[0][-3:]) > 0) & (int(file.split('.')[0][-3:]) < 21):
                        blenderFilePath = os.path.join(root, dir, file)
                        if 'Neutral' in blenderFilePath:
                            # if True:
                            pwd = 'galwaydnn'
                            cmd = blenderPath + " --background " \
                                  + blenderFilePath + " -P " + renderScript + " -- " + str(renderReal)

                            # print(cmd)
                            p = subprocess.call('echo {} | sudo -S {}'.format(pwd, cmd), shell=True)

                            c += 1

# # datetime object containing current date and time
# now = datetime.now()
#
# dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
# print("date and time =", dt_string)
