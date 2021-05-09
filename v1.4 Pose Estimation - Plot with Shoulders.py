import sys
import cv2
import os
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
        # Ensure you point to the correct path where models are located
        params["default_model_folder"] = dir_path + "/../../../models/"
        return params

def main():

        set_params()

        opWrapper = op.WrapperPython()

        #Constructing OpenPose object allocates GPU memory
        opWrapper.configure(params)

        opWrapper.start()


        #Opening OpenCV stream
        stream = cv2.VideoCapture(0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        t_start = time.time()

        neckLine = []
        eye_L = []
        eye_R = []

        shoulder = []

        count = 0

        while True:

                count +=1

                ret,img = stream.read()

                #cv2.imshow('Human Pose Estimation',img)

                # Output keypoints and the image with the human skeleton blended on it

                datum = op.Datum()
                datum.cvInputData = img
                opWrapper.emplaceAndPop([datum])

                neckLine.append([[datum.poseKeypoints[0][0][0], datum.poseKeypoints[0][1][0]],
                                 [datum.poseKeypoints[0][0][1], datum.poseKeypoints[0][1][1]],
                                 [datum.poseKeypoints[0][0][2], datum.poseKeypoints[0][1][2]]])

                eye_L.append([[datum.poseKeypoints[0][16][0]], [datum.poseKeypoints[0][16][1]], [datum.poseKeypoints[0][16][2]]])
                eye_R.append([[datum.poseKeypoints[0][15][0]], [datum.poseKeypoints[0][15][1]], [datum.poseKeypoints[0][15][2]]])

                # Left shoulder = point 5, right shoulder = point 2
                shoulder.append(
                    [[datum.poseKeypoints[0][5][0], datum.poseKeypoints[0][5][1], datum.poseKeypoints[0][5][2]],
                     [datum.poseKeypoints[0][2][0], datum.poseKeypoints[0][2][1], datum.poseKeypoints[0][2][2]]])
                cv2.imshow("OpenPose", datum.cvOutputData)

                key = cv2.waitKey(1)

                if key==ord('q') or count == 5: ######################################## COUNTER SET TO 20 FRAMES
                    break

        stream.release()
        #cv2.destroyAllWindows()

        neckLine = np.array(neckLine) # Convert array to NumPy array

        print(neckLine)

        data = {'nose_x': neckLine[:, 0, 0], 'nose_y': neckLine[:, 1, 0], 'nose_conf': neckLine[:, 2, 0],
                'neck_x': neckLine[:, 0, 1], 'neck_y': neckLine[:, 1, 1], 'neck_conf': neckLine[:, 2, 1]}

        df = pd.DataFrame(data)
        df.head()
        df_sma = pd.DataFrame(data)
        df_sma.head()

        # print(datum.poseKeypoints)

        # print(df_sma)

        # Moving average with specified window size
        window_size = 5
        for j in range(0, df.shape[1]):
            for i in range(0, df.shape[0] - window_size+1):
                df_sma.loc[df_sma.index[i + window_size-1], str(df_sma.columns.values[j])] = np.round(((df.iloc[i, j] + df.iloc[i + 1, j] + df.iloc[i + 2, j] + df.iloc[i + 3, j] + df.iloc[i + 4, j]) / window_size), 1)

        df_sma.head()

        # for j in range(0, df.shape[1]):
        #     for i in range(0, df.shape[0] - window_size+1):
        #         df1.loc[df.index[i + window_size-1], str(df1.columns.values[j])] = np.round(((df1.iloc[i, j] + df1.iloc[i + 1, j] + df1.iloc[i + 2, j] + df1.iloc[i + 3, j] + df1.iloc[i + 4, j]) / window_size), 1)
        #         # print(df['nose_x'])


        # print(df)
        # print("\n\n\n")
        # print(df_sma)

        shoulder = np.array(shoulder)
        shoulder_data = {'left_x': shoulder[:,0,0], 'right_x': shoulder[:,1,0],'left_y': shoulder[:,0,1], 'right_y': shoulder[:,1,1]}

        df_shoulder = pd.DataFrame(shoulder_data)
        df_shoulder.head()
        # print(df_shoulder)

        neck_x = data['neck_x']
        nose_x = data['nose_x']
        neck_y = data['neck_y']
        nose_y = data['nose_y']

        filename = "neck_data.csv"
        f = open(filename, "a+")
        f.write("\n ,")

        for i in range(len(neck_y)):

            if i == 0:
                f.write(str(np.mean(np.subtract(nose_y,neck_y))) + ", , ")

            # print(str(nose_y[i] - neck_y[i]) + ",  " + str(((nose_y[i] - neck_y[i])**2 + (nose_x[i] - neck_x[i])**2)**0.5))



            # f.write(str(nose_y[i] - neck_y[i]) + ", " + str(((nose_y[i] - neck_y[i])**2 + (nose_x[i] - neck_x[i])**2)**0.5) + "\n")
            f.write(str(nose_y[i] - neck_y[i]) + ", ")

        f.close()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # plt.figure(figsize=[15, 10])
        # plt.grid(True)
        # plt.plot(df['nose_x'], label='Nose')
        # plt.plot(df_sma['nose_x'], label='SMA-Nose: Smoothing Raw Values')
        fig, ax = plt.subplots()

        for i in range(0,len(neckLine)):
            if i == 0:
                ax.plot([df_sma['nose_x'][i], df_sma['neck_x'][i]], [df_sma['nose_y'][i], df_sma['neck_y'][i]], color="r", label="Neck to Nose Line")
                # ax.plot([df_sma['nose_x'][i], df_sma['neck_x'][i]], [df_sma['nose_y'][i], df_sma['neck_y'][i]], [df_sma['nose_z'][i], df_sma['neck_z'][i]], color="r", label="Neck to Nose Line")
                # ax.plot(df[i][0], df[i][1], df[i][2], color="r", label="Neck to Nose Line")
                ax.scatter(eye_L[i][0], eye_L[i][1], color="green", label="Left Eye")
                ax.scatter(eye_R[i][0], eye_R[i][1], color="b", label="Right Eye")

                ax.plot([df_shoulder['left_x'][i], df_sma['neck_x'][i]], [df_shoulder['left_y'][i], df_sma['neck_y'][i]], color="m", label="Left Shoulder")
                ax.plot([df_sma['neck_x'][i], df_shoulder['right_x'][i]], [df_sma['neck_y'][i], df_shoulder['right_y'][i]], color="darkorange", label="Right Shoulder")

            else:
                ax.plot([df_sma['nose_x'][i], df_sma['neck_x'][i]], [df_sma['nose_y'][i], df_sma['neck_y'][i]], color="r")
                # ax.plot(neckLine[i][0], neckLine[i][1], neckLine[i][2], color="r")
                ax.scatter(eye_L[i][0], eye_L[i][1], color="green")
                ax.scatter(eye_R[i][0], eye_R[i][1], color="b")
                ax.plot([df_shoulder['left_x'][i], df_sma['neck_x'][i]], [df_shoulder['left_y'][i], df_sma['neck_y'][i]], color="m")
                ax.plot([df_sma['neck_x'][i], df_shoulder['right_x'][i]], [df_sma['neck_y'][i], df_shoulder['right_y'][i]], color="darkorange")

        ax.plot(df_shoulder['left_x'][i], df_shoulder['left_y'][i], 'k-o')
        ax.plot(df_shoulder['right_x'][i], df_shoulder['right_y'][i], 'k-o')
        ax.plot(df_sma['neck_x'][i], df_sma['neck_y'][i], 'k-o')
        ax.plot(df_sma['nose_x'][i], df_sma['nose_y'][i], 'k-o')


        # ax.set_xlim(500,250)
        ax.set_ylim(500, 100)
        ax.set_title("Plot of Key Extracted Points")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_zlabel('Z Label')


        plt.legend()
        plt.show()




if __name__ == '__main__':
        main()