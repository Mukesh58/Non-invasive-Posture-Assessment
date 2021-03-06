import sys
import cv2
import os
from os import path
from sys import platform
import time
import numpy as np
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

def save_data(data, current_time):
    # Save dataFrame to file
    # t = time.localtime()
    # current_time = time.strftime("%m.%d %H%M", t)
    pickleName = check_fileName(str(current_time + "_Data.pkl"))
    data.to_pickle(pickleName)

    csvName = check_fileName(str(current_time + "_Data.csv"))
    data.to_csv(csvName)
    return pickleName

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

        # Write video file
        if write_flag == True:
            # Get current width of frame
            width = stream.get(3)
            height = stream.get(4)
            fps = stream.get(5)
            print('width: {}, height: {}, fps: {}'.format(width, height, fps))

            # Define the codec and create VideoWriter object
            fileName = str(start_time + '_Video.avi')

            fileName = check_fileName(fileName)

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            # (Filename, Codec, FPS, Width, height)
            out = cv2.VideoWriter(fileName, fourcc, 12, (int(width), int(height)))

        font = cv2.FONT_HERSHEY_SIMPLEX

        t_start = time.time()

        poseData = []
        calibData = []
        frameData = []
        neckLine = []
        eyes = []
        eyeR = []
        eyeL = []
        shoulder = []

        frameNum = 0 # Counter that monitors the labels the image frame captured via OpenCV
        calibCounter = 0 # Counter allowing for calibration sequence
        calibFlag = False # Allows calibration sequence to run

        print("[Frame_Number, [Nose, Neck], [[eyeL], [eyeR]], [[shoulderL], [shoulderR]]")



        while True:

                frameNum +=1
                print(frameNum)

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
                if frameNum % 1 == 0  and frameNum > 0 * fps:

                    #cv2.imshow('Human Pose Estimation',img)

                    # Output keypoints and the image with the human skeleton blended on it

                    # neckLine = [[Nose_x, Nose_y, conf], [Neck_x, Neck_y, conf]]
                    try:
                        neckLine = [[datum.poseKeypoints[0][0][0], datum.poseKeypoints[0][0][1], datum.poseKeypoints[0][0][2]],
                                     [datum.poseKeypoints[0][1][0], datum.poseKeypoints[0][1][1], datum.poseKeypoints[0][1][2]]]

                        boolUserInFrame = True

                    except:
                        neckLine = [[-1, -1, -1], [-1, -1, -1]]

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

                    # print("user: " + str(boolUserInFrame))
                    # print("calib: " + str(calibFlag))

                    # Branch runs if user is in frame and program not in calibration sequence
                    if boolUserInFrame == True and calibFlag == False:

                        frameData = [[frameNum,0,0], neckLine[0], neckLine[1], eyeL, eyeR, shoulder[0], shoulder[1]]
                        print(frameData)

                        # Data Structure: [Frame_Number, [Nose], [Neck], [eyeL], [eyeR], [shoulderL], [shoulderR]]
                        poseData.append(frameData)

                    # Branch runs if user is in frame and program in initial calibration sequence
                    elif boolUserInFrame == True and calibFlag == True:

                        frameData = [[calibCounter, 0, 0], neckLine[0], neckLine[1], eyeL, eyeR, shoulder[0], shoulder[1]]

                        print(frameData[1][2])
                        print(frameData[2][2])


                        # Only include frame if confidence exceeds required threshold.

                        if(frameData[1][2] > 0.8 and frameData[2][2] > 0.55):

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

                            calib_nose_x = calibData[:, 1]
                            calib_nose_y = calibData[:, 1]
                            calib_neck_x = calibData[:, 2]
                            calib_neck_y = calibData[:, 2]
                            calib_eyeL_x = calibData[:, 3]
                            calib_eyeL_y = calibData[:, 3]
                            calib_eyeR_x = calibData[:, 4]
                            calib_eyeR_y = calibData[:, 4]
                            calib_shoulderL_x = calibData[:, 5]
                            calib_shoulderL_y = calibData[:, 5]
                            calib_shoulderR_x = calibData[:, 6]
                            calib_shoulderR_y = calibData[:, 6]

                            # else:
                            # print("Other")

                    img = cv2.flip(datum.cvOutputData, 1)

                    cv2.imshow("OpenPose", img)

                    key = cv2.waitKey(1)

                    if key==ord('q') or frameNum == 1000: ######################################## COUNTER SET TO 25 FRAMES
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
        pickleName = save_data(df_sma, start_time)

        output = pd.read_pickle(pickleName)
        print(output)

        # output = pd.read_csv(csvName)
        # print(output)

        neck_x = data['neck_x']
        nose_x = data['nose_x']
        neck_y = data['neck_y']
        nose_y = data['nose_y']

        filename = "neck_data.csv"
        f = open(filename, "a+")
        f.write("\n ,")

        # Write values of the vertical distance between nose and neck
        for i in range(len(neck_y)):

            if i == 0:
                # Write the mean difference value for this run
                f.write(str(np.mean(np.subtract(nose_y,neck_y))) + "\n\n")

            # print(str(nose_y[i] - neck_y[i]) + ",  " + str(((nose_y[i] - neck_y[i])**2 + (nose_x[i] - neck_x[i])**2)**0.5))

            # f.write(str(nose_y[i] - neck_y[i]) + ", " + str(((nose_y[i] - neck_y[i])**2 + (nose_x[i] - neck_x[i])**2)**0.5) + "\n")
            f.write(str(nose_y[i] - neck_y[i]) + "\n")

        f.close()

        #---------------- Plotting of the data points ------------------------------------

#         xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx



if __name__ == '__main__':
        main()