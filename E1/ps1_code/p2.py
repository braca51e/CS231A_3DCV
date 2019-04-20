# CS231A Homework 1, Problem 2
import numpy as np

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    # TODO: Fill in this code
    real_XY = np.append(np.array(real_XY), np.ones((len(real_XY), 1)), axis=1)
    front_image = np.append(np.array(front_image), np.zeros((len(front_image), 1)), axis=1)
    front_image = np.append(np.array(front_image), np.ones((len(front_image), 1)), axis=1)
    back_image = np.append(np.array(back_image), 150*np.ones((len(back_image), 1)), axis=1)
    back_image = np.append(np.array(back_image), np.ones((len(back_image), 1)), axis=1)
    A_ = np.vstack((front_image, back_image))
    b = np.vstack((real_XY, real_XY))
    b = b.flatten()
    A = np.zeros((3*len(A_), 12))

    for i in range(0, len(A), 3):
        A[i, :4] = A_[int(i/3), :]
        A[i+1, 4:8] = A_[int(i/3), :]
        A[i+2, 8:] = A_[int(i/3), :]
    #Solve Linear Systems
    m = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b)

    return m.reshape(3, 4)

'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    #TODO: Fill in this code
    real_XY = np.append(np.array(real_XY), np.ones((len(real_XY), 1)), axis=1)
    front_image = np.append(np.array(front_image), np.zeros((len(front_image), 1)), axis=1)
    front_image = np.append(np.array(front_image), np.ones((len(front_image), 1)), axis=1)
    back_image = np.append(np.array(back_image), 150*np.ones((len(back_image), 1)), axis=1)
    back_image = np.append(np.array(back_image), np.ones((len(back_image), 1)), axis=1)
    points_a = np.vstack((front_image, back_image))
    points_a = np.dot(camera_matrix, points_a.T).T
    points_b =np.vstack((real_XY, real_XY))

    return np.sqrt(np.sum((points_a - points_b)/len(points_a)))
        
if __name__ == '__main__':
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))
