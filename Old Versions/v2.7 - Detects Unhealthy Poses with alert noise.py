import sys
import cv2
import os
from os import path
from sys import platform
import time
import numpy as np
import winsound
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d

dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append('usr/local/python');

params = dict()
params["model_folder"] = "../../models/"

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../python/openpose/Release')
            os.add_dll_directory(dir_path + '/../x64/Release')
            os.add_dll_directory(dir_path + '/../bin')
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

except Exception as e:
    print(e)
    sys.exit(-1)

def set_params():

        params = dict()
        params["logging_level"] = 3
        params["output_resolution"] = "-1x-1"
        params["net_resolution"] = "-1x368"
        params["model_pose"] = "BODY_25"
        params["alpha_pose"] = 0.6
        params["scale_gap"] = 0.3
        params["scale_number"] = 1
        params["render_threshold"] = 0.05
        # If GPU version is built, and multiple GPUs are available, set the ID here
        params["num_gpu_start"] = 0
        params["disable_blending"] = False
        params["write_video"] = dir_path + "/file.avi"
        # Ensure you point to the correct path where models are located
        params["default_model_folder"] = dir_path + "/../../../models/"
        return params

# Used to increment filename for saving video files
def check_fileName(fPath):
    if path.exists(fPath):
        n = 1
        while True:
            filePath = "{0}_{2}{1}".format(*path.splitext(fPath) + (n,))
            if path.exists(filePath):
                n += 1
            else:
                return filePath
    else:
        return fPath

def save_data(data, current_time, file_tag):
    # Save dataFrame to file
    # t = time.localtime()
    # current_time = time.strftime("%m.%d %H%M", t)
    # pickleName = check_fileName(str("data/" + current_time + "_Data.pkl"))
    # data.to_pickle(pickleName)

    csvName = check_fileName(str("data/" + current_time + "_Data" + file_tag + ".csv"))
    data.to_csv(csvName)
    return csvName

def main():

        set_params()

        t = time.localtime()
        start_time = time.strftime("%y.%m.%d %H%M", t)

        opWrapper = op.WrapperPython()

        #Constructing OpenPose object allocates GPU memory
        opWrapper.configure(params)
        opWrapper.start()

        #Opening OpenCV stream
        stream = cv2.VideoCapture(0)

        write_flag = True    # Write video file
        file_tag = ""

        # Write video file
        if write_flag == True:
            # Get current width of frame
            width = stream.get(3)
            height = stream.get(4)
            fps = stream.get(5)
            print('width: {}, height: {}, fps: {}'.format(width, height, fps))

            # Define the codec and create VideoWriter object
            fileName = str("data/" + start_time + '_Video' + file_tag + '.avi' )

            fileName = check_fileName(fileName)

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            # (Filename, Codec, FPS, Width, height)
            out = cv2.VideoWriter(fileName, fourcc, 12, (int(width), int(height)))

        font = cv2.FONT_HERSHEY_SIMPLEX

        t_start = time.time()

        poseData = []
        calibData = []
        frameData = []
        noseNeck = []
        eyes = []
        eyeR = []
        eyeL = []
        shoulder = []

        frameNum = 0 # Counter that monitors the labels the image frame captured via OpenCV
        calibCounter = 0 # Counter allowing for calibration sequence
        count_NoUser = 0 # Count the frames without user detected
        count_User = 0 # Counter for frames with a user detected after user not in frame for n frames
        calibFlag = True # Allows calibration sequence to run

        print("[Frame_Number, [Nose, Neck], [[eyeL], [eyeR]], [[shoulderL], [shoulderR]]")



        while True:

                frameNum +=1
                # print(frameNum)

                boolUserInFrame = False

                ret,img = stream.read()

                if ret == True:
                    # Handles the mirroring of the current frame

                    datum = op.Datum()
                    datum.cvInputData = img
                    opWrapper.emplaceAndPop([datum])

                    # Write video file
                    if write_flag == True:
                        img = cv2.flip(datum.cvOutputData, 1)
                        # Saves for video
                        out.write(img)

                # Capture every nth frame and only begin after a t second initialisation time to give user time to settle
                if frameNum % 3 == 0  and frameNum > 0 * fps:

                    #cv2.imshow('Human Pose Estimation',img)

                    # Output keypoints and the image with the human skeleton blended on it

                    # noseNeck = [[Nose_x, Nose_y, conf], [Neck_x, Neck_y, conf]]
                    try:
                        noseNeck = [[datum.poseKeypoints[0][0][0], datum.poseKeypoints[0][0][1], datum.poseKeypoints[0][0][2]],
                                     [datum.poseKeypoints[0][1][0], datum.poseKeypoints[0][1][1], datum.poseKeypoints[0][1][2]]]

                        boolUserInFrame = True

                    except:
                        noseNeck = [[-1, -1, -1], [-1, -1, -1]]

                    # Left Eye: eyeL = [eyeL_x, eyeL_y, conf]
                    try:
                        eyeL = [datum.poseKeypoints[0][16][0], datum.poseKeypoints[0][16][1], datum.poseKeypoints[0][16][2]]

                        boolUserInFrame = True
                    except:
                        eyeL = [-1, -1, -1]

                    # Right Eye: eyeR = [eyeR_x, eyeR_y, conf]
                    try:
                        eyeR = [datum.poseKeypoints[0][15][0], datum.poseKeypoints[0][15][1], datum.poseKeypoints[0][15][2]]

                        boolUserInFrame = True
                    except:
                        eyeR = [-1, -1, -1]

                    # Shoulders: Left shoulder = point 5, right shoulder = point 2
                    # shoulder = [[shoulderL_x, shoulderL_y, conf], [shoulderR_x, shoulderR_y, conf]]
                    try:
                        shoulder = [[datum.poseKeypoints[0][5][0], datum.poseKeypoints[0][5][1], datum.poseKeypoints[0][5][2]],
                         [datum.poseKeypoints[0][2][0], datum.poseKeypoints[0][2][1], datum.poseKeypoints[0][2][2]]]

                        boolUserInFrame = True
                    except:
                        shoulder = [[-1, -1, -1], [-1, -1, -1]]


                    # Check if user has been out of frame for an extended period of time
                    # If so, recalibrate
                    if boolUserInFrame == False and calibFlag == False:
                        count_NoUser += 1
                        if count_NoUser > 10:
                            count_User = 0
                    elif boolUserInFrame == True and calibFlag == False:
                        count_User += 1
                        if count_User > 10:
                            count_NoUser = 0

                    # print("User: " + str(count_User))
                    # print("No User: " + str(count_NoUser))

                    if count_NoUser > 20:
                        count_NoUser = 0
                        calibData = []
                        calibCounter = 0
                        calibFlag = True
                        print("Recalibration commenced)")



                    # Branch runs if user is in frame and program not in calibration sequence
                    if boolUserInFrame == True and calibFlag == False:

                        frameData = [[frameNum,0,0], noseNeck[0], noseNeck[1], eyeL, eyeR, shoulder[0], shoulder[1]]
                        print(frameData)

                        # Data Structure: [Frame_Number, [Nose], [Neck], [eyeL], [eyeR], [shoulderL], [shoulderR]]
                        poseData.append(frameData)

                        diff_nose_x = frameData[1][0] - calib_nose[0]
                        diff_nose_y = frameData[1][1] - calib_nose[1]
                        diff_neck_x = frameData[2][0] - calib_neck[0]
                        diff_neck_y = frameData[2][1] - calib_neck[1]
                        diff_eyeL_x = frameData[3][0] - calib_eyeL[0]
                        diff_eyeL_y = frameData[3][1] - calib_eyeL[1]
                        diff_eyeR_x = frameData[4][0] - calib_eyeR[0]
                        diff_eyeR_y = frameData[4][1] - calib_eyeR[1]
                        diff_shoulderL_x = frameData[5][0] - calib_shoulderL[0]
                        diff_shoulderL_y = frameData[5][1] - calib_shoulderL[1]
                        diff_shoulderR_x = frameData[6][0] - calib_shoulderR[0]
                        diff_shoulderR_y = frameData[6][1] - calib_shoulderR[1]

                        dist_noseNeck_x = abs(frameData[1][0] - frameData[2][0])
                        dist_noseNeck_y = abs(frameData[1][1] - frameData[2][1])
                        dist_eyeLNose_y = abs(frameData[3][1] - frameData[1][1])
                        dist_eyeRNose_y = abs(frameData[4][1] - frameData[1][1])
                        dist_eyes_x = abs(frameData[3][0] - frameData[4][0])
                        dist_shoulderBreadth = abs(frameData[5][0] - frameData[6][0])

                        #  Left minus Right: If >0, Left eye is lower than right
                        offset_eyesLR_y = frameData[3][1] - frameData[4][1]

                        boolAbnormalPosture = False

                        print("Diff_nose_x: " + str(diff_nose_x))
                        print("Diff_eyeL_x: " + str(diff_eyeL_x))
                        print("Diff_eyeR_x: " + str(diff_eyeR_x))

                        # Detect Forward Head
                        # Distances: dist_noseNeck_y > 25.1825 and dist_eyeLNose_y < 6.0821 and dist_eyes_x >5.2467
                        if diff_nose_y > 35.30942 and diff_eyeL_y > 25 and diff_eyeR_y > 25 and diff_shoulderL_y > 5 and diff_shoulderR_y > 5:
                            print("Forward Head")
                            boolAbnormalPosture = True
                        elif diff_nose_y > 30.8454 and diff_eyeL_y > 22.1793 and diff_eyeR_y > 22.1793 and diff_neck_y > 17.0192 and diff_shoulderL_y > 18.5904 and diff_shoulderR_y > 18.5904:
                            print("Shoulder Slouch")
                            boolAbnormalPosture = True
                        elif diff_eyeL_y > 8.5414 and diff_eyeR_y < 6 and offset_eyesLR_y > 13.5396:
                            print("Head Tilt L")
                            boolAbnormalPosture = True
                        elif diff_eyeL_y < 6 and diff_eyeR_y > 8.5414 and offset_eyesLR_y < -10.0690:
                            print("Head Tilt R")
                            boolAbnormalPosture = True
                        elif diff_nose_x > 30.931 and diff_eyeL_x > 22.58788 and diff_eyeR_x > 22.58788:
                            print("Head Rotate L")
                            boolAbnormalPosture = True
                        elif diff_nose_x < -30.4166 and diff_eyeL_x < -24.527 and diff_eyeR_x < -24.527:
                            print("Head Rotate R")
                            boolAbnormalPosture = True
                        else:
                            pass

                        if boolAbnormalPosture:
                            winsound.Beep(440, 500)



                    # Branch runs if user is in frame and program in initial calibration sequence
                    elif boolUserInFrame == True and calibFlag == True:

                        frameData = [[calibCounter, 0, 0], noseNeck[0], noseNeck[1], eyeL, eyeR, shoulder[0], shoulder[1]]

                        print(frameData[1][2])
                        print(frameData[2][2])


                        # Only include frame if confidence exceeds required threshold.

                        if(frameData[1][2] > 0.8 and frameData[2][2] > 0.55):

                            print("Added to calib")

                            calibCounter += 1

                            # Data Structure: [Frame_Number, [Nose], [Neck], [eyeL], [eyeR], [shoulderL], [shoulderR]]
                            calibData.append(frameData)
                        print(frameData)



                        # End calibration sequence once desired frame count has been reached
                        if calibCounter >= 15:

                            calibFlag = False
                            print("********** CALIBRATION COMPLETE ********** ")

                            # Calculate Calibration Parameters
                            calibData = np.array(calibData)  # Convert array to NumPy array

                            calibLen = len(calibData)
                            print("Calib len = " + str(calibLen))

                            # Setting calibration Parameters

                            calibMean = np.mean(calibData[:, 1], axis=0)
                            calib_nose = [calibMean[0], calibMean[1]]
                            calibMean = np.mean(calibData[:, 2], axis=0)
                            calib_neck = [calibMean[0], calibMean[1]]
                            calibMean = np.mean(calibData[:, 3], axis=0)
                            calib_eyeL = [calibMean[0], calibMean[1]]
                            calibMean = np.mean(calibData[:, 4], axis=0)
                            calib_eyeR = [calibMean[0], calibMean[1]]
                            calibMean = np.mean(calibData[:, 5], axis=0)
                            calib_shoulderL = [calibMean[0], calibMean[1]]
                            calibMean = np.mean(calibData[:, 6], axis=0)
                            calib_shoulderR = [calibMean[0], calibMean[1]]

                            print(str(calib_nose[0]) + ", " + str(calib_nose[0]))

                            calib_noseNeck_x = abs(calibData[1][0] - calibData[2][0])
                            calib_noseNeck_y = abs(calibData[1][1] - calibData[2][1])
                            calib_eyeLNose_y = abs(calibData[3][1] - calibData[1][1])
                            calib_eyeRNose_y = abs(calibData[4][1] - calibData[1][1])
                            calib_eyes_x = abs(calibData[3][0] - calibData[4][0])
                            calib_shoulderBreadth = abs(calibData[5][0] - calibData[6][0])

                            #  Left minus Right: If >0, Left eye is lower than right
                            calib_offset_eyesLR_y = calibData[3][1] - calibData[4][1]


                            print("********** CALIBRATION COMPLETE ********** \n")

                            # else:
                            # print("Other")


                    img = cv2.flip(datum.cvOutputData, 1)

                    cv2.imshow("OpenPose", img)

                    # if frameNum % 50 == 0:
                    #     print("--------------------Image Saving-----------------")
                    #     img_name = str("img/" + start_time + "_" + str(frameNum) + "_Data.png")
                    #     cv2.imwrite(img_name, img)


                    key = cv2.waitKey(1)

                    if key==ord('q') or frameNum == 1000: ######################################## COUNTER SET TO 25 FRAMES
                        print("--------------------Image Saving-----------------")
                        img_name = str("img/" + start_time + "_Data.png")
                        cv2.imwrite(img_name, img)
                        break

        stream.release()
        #cv2.destroyAllWindows()

        poseData = np.array(poseData) # Convert array to NumPy array

        # poseData Structure: [Frame_Number, [Nose], [Neck], [eyeL], [eyeR], [shoulderL], [shoulderR]]

        data_nose = poseData[:, 1]
        data_neck = poseData[:, 2]
        eyeL = poseData[:, 3]
        eyeR = poseData[:, 4]
        shoulderL = poseData[:, 5]
        shoulderR = poseData[:, 6]

        data = {'nose_x': data_nose[:, 0], 'nose_y': data_nose[:, 1], 'nose_conf': data_nose[:, 2],
                'neck_x': data_neck[:, 0], 'neck_y': data_neck[:, 1], 'neck_conf': data_neck[:, 2],
                'eyeL_x': eyeL[:, 0], 'eyeL_y': eyeL[:, 1], 'eyeL_conf': eyeL[:, 2],
                'eyeR_x': eyeR[:, 0], 'eyeR_y': eyeR[:, 1], 'eyeR_conf': eyeR[:, 2],
                'shoulderL_x': shoulderL[:, 0], 'shoulderL_y': shoulderL[:, 1], 'shoulderL_conf': shoulderL[:, 2],
                'shoulderR_x': shoulderR[:, 0], 'shoulderR_y': shoulderR[:, 1], 'shoulderR_conf': shoulderR[:, 2],
                }

        df = pd.DataFrame(data)
        #********************* Access first n lines with df.head(n) ****************
        # print(df)

        df_sma = pd.DataFrame(data)

        # print(datum.poseKeypoints)
        # print(df_sma)

        # Moving average with specified window size
        window_size = 5
        for j in range(0, df.shape[1]):
            for i in range(0, df.shape[0] - window_size+1):
                df_sma.loc[df_sma.index[i + window_size-1], str(df_sma.columns.values[j])] = np.round(((df.iloc[i, j] + df.iloc[i + 1, j] + df.iloc[i + 2, j] + df.iloc[i + 3, j] + df.iloc[i + 4, j]) / window_size), 6)

        # print(df_sma)

        #****** Save Data to file *******
        # pickleName = check_fileName("a_file.pkl")
        # df_sma.to_pickle(pickleName)
        # pickleName = save_data(df_sma, start_time, file_tag)

        # output = pd.read_pickle(pickleName)
        # print(output)

        csvName = save_data(df_sma, start_time, file_tag)
        output = pd.read_csv(csvName)
        print(output)

        neck_x = data['neck_x']
        nose_x = data['nose_x']
        neck_y = data['neck_y']
        nose_y = data['nose_y']

        # filename = "neck_data.csv"
        # f = open(filename, "a+")
        # f.write("\n ,")
        #
        # # Write values of the vertical distance between nose and neck
        # for i in range(len(neck_y)):
        #
        #     if i == 0:
        #         # Write the mean difference value for this run
        #         f.write(str(np.mean(np.subtract(nose_y,neck_y))) + "\n\n")
        #
        #     # print(str(nose_y[i] - neck_y[i]) + ",  " + str(((nose_y[i] - neck_y[i])**2 + (nose_x[i] - neck_x[i])**2)**0.5))
        #
        #     # f.write(str(nose_y[i] - neck_y[i]) + ", " + str(((nose_y[i] - neck_y[i])**2 + (nose_x[i] - neck_x[i])**2)**0.5) + "\n")
        #     f.write(str(nose_y[i] - neck_y[i]) + "\n")
        #
        # f.close()

        #---------------- Plotting of the data points ------------------------------------

#         xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx



if __name__ == '__main__':
        main()