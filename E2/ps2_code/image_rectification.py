import numpy as np
import matplotlib.pyplot as plt
from fundamental_matrix_estimation import *

'''
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points1)^T * F * points2 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(points1, points2, F):
    # TODO: Implement this method!
    #compute epipole e
    e_lines = np.dot(F.T, points2.T).T
    #Find SVD 
    U, S, V_T = np.linalg.svd(e_lines, full_matrices=True)
    e = (V_T[-1:, :])
    #bring back to homegeneus 
    e /= e[0,2]
    return e
    
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    # TODO: Implement this method!
    e2 = np.squeeze(e2)
    _, width = im2.shape
    height, _ = im2.shape
    T = np.array([[1, 0, -width/2],
                  [0, 1, -height/2],
                  [0, 0, 1]])
    T_inv  = np.linalg.inv(T)
    e2_ = np.dot(T, e2.T)
    e2_ = e2_/e2_[2]
    #get alpha
    if(e2[0] >= 0):
        alpha = 1
    else:
        alpha = -1

    R = np.array([[alpha*(e2_[0]/np.sqrt(e2_[0]**2+e2_[1]**2)), alpha*(e2_[1]/np.sqrt(e2_[0]**2+e2_[1]**2)),0],
                  [-alpha*(e2_[1]/np.sqrt(e2_[0]**2+e2_[1]**2)), alpha*(e2_[0]/np.sqrt(e2_[0]**2+e2_[1]**2)), 0],
                  [0, 0, 1]])
    e2_ = np.dot(R, e2_)
    
    G = np.array([[1, 0, 0], 
                  [0, 1, 0],
                  [-1/e2_[0], 0, 1]])

    H2 = np.dot(np.dot(np.dot(T_inv, G), R), T)
    #Compute M
    e2_x = np.array([[0, -e2[2], e2[2]],
                     [e2[2], 0, -e2[0]],
                     [-e2[1], e2[0], 0]])

    M = np.dot(e2_x, F) + np.dot(e2.reshape(3,1), np.array([[1, 1, 1]]))
    #Get p_hats
    p_hat = np.dot(np.dot(H2, M), points1.T).T
    p_hat_ = np.dot(H2, points2.T).T
    W = np.array(list(map(lambda x: x/x[2], p_hat)))
    p_hat_ = np.array(list(map(lambda x: x/x[2], p_hat_)))
    b = p_hat_[:,0]
    a = np.dot(np.dot(np.linalg.inv(np.dot(W.T, W)), W.T), b)
    Ha = np.array([[a[0], a[1], a[2]],
                     [0, 1, 0],
                     [0, 0, 1]])
    H1 = np.dot(Ha, np.dot(H2, M))

    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print
    print("H2:\n", H2)

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
