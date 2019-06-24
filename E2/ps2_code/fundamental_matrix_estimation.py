import numpy as np
#from scipy.misc import imread
from scipy.misc.pilutil import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    #raise Exception('Not Implemented Error')
    #Create matrix Ws
    W = np.zeros((len(points1), 9))
    for k, p in enumerate(points1):
        u1, v1, _ = points1[k]
        u1_, v1_, _ = points2[k]
        W[k] = np.array([u1*u1_, v1*u1_, u1_, u1*v1_, v1*v1_, v1_, u1, v1, 1])

    U, S, V_T = np.linalg.svd(W, full_matrices=True)
    F = (V_T[-1, :]).reshape((3,3))

    U, S, V_T = np.linalg.svd(F, full_matrices=True)
    S = np.diag(S)
    #enforce property that essential matrix has only 2 non-zero eigenvalues
    S[-1, :] = 0.0
    F_ = np.dot(U, np.dot(S, V_T))

    return F_

    
'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''

def normalized_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    #raise Exception('Not Implemented Error')
    
    points1_ = points1[:, 0:2]
    points2_ = points2[:, 0:2]

    #Find centroid
    centroid_p1 = np.average(points1_, axis=0)
    centroid_p2 = np.average(points2_, axis=0)

    #Center points
    points1_centered = points1_ - centroid_p1
    points2_centered = points2_ - centroid_p2
    #Compute distance to center 
    points1_distance = np.sqrt(np.sum(points1_centered**2, axis=1))
    points2_distance = np.sqrt(np.sum(points2_centered**2, axis=1))
    mean1_dist_origin = np.mean(points1_distance)
    mean2_dist_origin = np.mean(points2_distance)
    #scaling factor should be 2/mean_distance to center
    s1 = 2/mean1_dist_origin
    s2 = 2/mean2_dist_origin
    
    T = np.array([[s1, 0, -s1*centroid_p1[0]], 
                   [0, s1, -s1*centroid_p1[1]], 
                   [0, 0, 1]])
    
    T_ = np.array([[s2, 0, -s2*centroid_p2[0]], 
                   [0, s2, -s2*centroid_p2[1]], 
                   [0, 0, 1]])

    #Apply transform
    q1 = np.transpose(np.dot(T, points1.T))
    q2 = np.transpose(np.dot(T_, points2.T))

    F_q = lls_eight_point_alg(q1, q2)

    #Undo normalization
    F = np.dot(np.dot(T_.T, F_q), T)
    
    return F

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    # TODO: Implement this method!
    #raise Exception('Not Implemented Error')
    plt.subplot(1,2,1)
    h, w = im1.shape
    ep_lines = np.dot(F.T, points2.T).T

    for k, ln1 in enumerate(ep_lines):
        a, b, c = ln1
        m = -(a*1.0/b)
        y_inter = -(c*1.0/b)
        plt.plot([0, w], [y_inter, m*h + y_inter], 'r')
        plt.plot(points1[k][0], points1[k][1], 'b*')
    plt.imshow(im1, cmap='gray')
    
    plt.subplot(1,2,2)
    h, w = im2.shape
    ep_lines = np.dot(F, points1.T).T
    for k, ln2 in enumerate(ep_lines):
        a, b, c = ln2
        m = -(a*1.0/b)
        y_inter = -(c*1.0/b)
        plt.plot([0, w], [y_inter, m*h + y_inter], 'r')
        plt.plot(points2[k][0], points2[k][1], 'b*')
    plt.imshow(im2, cmap='gray')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # TODO: Implement this method!
    #raise Exception('Not Implemented Error')
    #lines of points1
    ep_lines = np.dot(F.T, points2.T).T
    error = []

    for k, ln1 in enumerate(ep_lines):
        a, b, c = ln1
        x, y, _ = points1[k]
        error.append([abs(a*x + b*y + 1.0*c)])

    ep_lines = np.dot(F, points1.T).T
    for k, ln2 in enumerate(ep_lines):
        a, b, c = ln2
        x, y, _ = points2[k]
        error.append([abs(a*x + b*y + 1.0*c)])

    return np.array(error).mean()

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
