import sys
import cv2
import os
from sys import platform
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append('usr/local/python');

params = dict()
params["model_folder"] = "../../models/"

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
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

        x = []
        y = []
        z = []
        x1 = []
        y1 = []
        z1 = []

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        datasets = []
        eye_L = []
        eye_R = []

        while True:

                ret,img = stream.read()

                #cv2.imshow('Human Pose Estimation',img)

                # Output keypoints and the image with the human skeleton blended on it
                # keypoints, output_image = opWrapper.forwardPass(img, True)
                #time.sleep(0.5)

                datum = op.Datum()
                # imageToProcess = cv2.imread(img)
                datum.cvInputData = img
                opWrapper.emplaceAndPop([datum])

                #print("Body keypoints: \n" + str(datum.poseKeypoints))
                # print("Time: " + str(time.time() - t_start))
                # print(str(datum.poseKeypoints[0][0]) + "\n" + str(datum.poseKeypoints[0][1]))

                x.append(datum.poseKeypoints[0][0][0])
                y.append(datum.poseKeypoints[0][0][1])
                z.append(datum.poseKeypoints[0][0][2])

                x1.append(datum.poseKeypoints[0][1][0])
                y1.append(datum.poseKeypoints[0][1][1])
                z1.append(datum.poseKeypoints[0][1][2])

                datasets.append([[datum.poseKeypoints[0][0][0],datum.poseKeypoints[0][1][0]], [datum.poseKeypoints[0][0][1],datum.poseKeypoints[0][1][1]], [datum.poseKeypoints[0][0][2],datum.poseKeypoints[0][1][2]]])
                eye_L.append([[datum.poseKeypoints[0][16][0]], [datum.poseKeypoints[0][16][1]], [datum.poseKeypoints[0][16][2]]])
                eye_R.append([[datum.poseKeypoints[0][15][0]], [datum.poseKeypoints[0][15][1]], [datum.poseKeypoints[0][15][2]]])

                # Display the stream
                # cv2.putText(img,'OpenPose using Python-OpenCV',(20,30), font, 1,(255,255,255),1,cv2.LINE_AA)

                cv2.imshow("OpenPose", datum.cvOutputData)

                key = cv2.waitKey(1)

                if key==ord('q'):
                    break

        stream.release()
        #cv2.destroyAllWindows()

        for i in range(len(datasets)):
            if i == 0:
                ax.plot(datasets[i][0], datasets[i][1], datasets[i][2], color="r", label="Neck to Nose Line")
                ax.scatter(eye_L[i][0], eye_L[i][1], eye_L[i][2], color="green", label="Left Eye")
                ax.scatter(eye_R[i][0], eye_R[i][1], eye_R[i][2], color="blue", label="Right Eye")
            else:
                ax.plot(datasets[i][0], datasets[i][1], datasets[i][2], color="r")
                ax.scatter(eye_L[i][0], eye_L[i][1], eye_L[i][2], color="green")
                ax.scatter(eye_R[i][0], eye_R[i][1], eye_R[i][2], color="blue")

        #ax.scatter(x, y, z, c='r', marker='o')
        #ax.scatter(x1, y1, z1, c='b', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.legend()

        plt.show()


        # Plotting data points
        # ax.plot(dataset["x"], dataset["y"], dataset["z"], color="r", label="Neck to Nose Line")
        # ax.scatter(eye_L[i][0], eye_L[i][1], eye_L[i][2], color="green", label="Left Eye")
        # ax.scatter(eye_R[i][0], eye_R[i][1], eye_R[i][2], color="blue", label="Right Eye")

        # Appending data points in loop
        # datasets.append({"x": [datum.poseKeypoints[0][0][0], datum.poseKeypoints[0][1][0]],
        #                  "y": [datum.poseKeypoints[0][0][1], datum.poseKeypoints[0][1][1]],
        #                  "z": [datum.poseKeypoints[0][0][2], datum.poseKeypoints[0][1][2]]})
        # eye_L.append(
        #     [[datum.poseKeypoints[0][16][0]], [datum.poseKeypoints[0][16][1]], [datum.poseKeypoints[0][16][2]]])
        # eye_R.append(
        #     [[datum.poseKeypoints[0][15][0]], [datum.poseKeypoints[0][15][1]], [datum.poseKeypoints[0][15][2]]])


if __name__ == '__main__':
        main()