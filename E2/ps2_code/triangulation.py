import sys
import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # TODO: Implement this method!
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    U, D, V_T = np.linalg.svd(E, full_matrices=True)
    Q1 = np.dot(U, np.dot(W, V_T))
    Q2 = np.dot(U, np.dot(W.T, V_T))
    R1 = np.dot(np.linalg.det(Q1), Q1)
    R2 = np.dot(np.linalg.det(Q2), Q2)
    #third colum U
    T1 = U[:, 2]
    T2 = -U[:, 2]

    RT = np.zeros((4, 3, 4))
    RT1 = np.hstack((R1, T1.reshape(3,1)))
    RT[0,:,:] = RT1
    RT2 = np.hstack((R1,T2.reshape(3,1)))
    RT[1,:,:] = RT2
    RT3 = np.hstack((R2,T1.reshape(3,1)))
    RT[2,:,:] = RT3
    RT4 = np.hstack((R2,T2.reshape(3,1)))
    RT[3,:,:] = RT4
    return RT
'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    A = np.zeros((len(camera_matrices)*len(image_points), 4))
    
    for k, point in enumerate(image_points):
        M = camera_matrices[k]
        m1 = M[0, :]
        m2 = M[1, :]
        m3 = M[2, :]
        x = point[0]
        y = point[1]
        A[2*k] = np.array([x*m3-m1])
        A[2*k+1] = np.array([y*m3-m2])
    U, S, V_T = np.linalg.svd(A, full_matrices=True)
    #Take last row of V_T homogeniues
    P = V_T[-1]
    P /= P[-1]
    return P[0:3]

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # TODO: Implement this method!
    error = np.zeros((2*len(camera_matrices), 1))
    point_3d = np.concatenate((point_3d,np.array([1])),axis=0)
    for k, point in enumerate(image_points):
        M = camera_matrices[k]
        p_ = np.dot(M, point_3d)
        p_ /= p_[2]
        error[2*k:2*k+2, :] = (p_[0:2] - image_points[k]).reshape((2,1))

    return error

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # TODO: Implement this method!
    J = np.zeros((2*len(camera_matrices), 3))
    point_3d = np.hstack((point_3d, 1))
    #Apply quatient rule pi = [m0P/m2P m1P/m2P].T
    for k, M in enumerate(camera_matrices):
        pi = M.dot(point_3d)
        #compute for x
        J[2*k, 0] = (pi[2]*M[0,0] - pi[0]*M[2,0])/pi[2]**2
        J[2*k, 1] = (pi[2]*M[0,1] - pi[0]*M[2,1])/pi[2]**2
        J[2*k, 2] = (pi[2]*M[0,2] - pi[0]*M[2,2])/pi[2]**2
        #compute for y
        J[2*k+1, 0] = (pi[2]*M[1,0] - pi[1]*M[2,0])/pi[2]**2
        J[2*k+1, 1] = (pi[2]*M[1,1] - pi[1]*M[2,1])/pi[2]**2
        J[2*k+1, 2] = (pi[2]*M[1,2] - pi[1]*M[2,2])/pi[2]**2

    return J

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    #part a good stimate
    P_ = linear_estimate_3d_point(image_points, camera_matrices)
    #part b 10 iterations
    for i in range(10):
        J = jacobian(P_, camera_matrices)
        e = reprojection_error(P_, image_points, camera_matrices)
        P_ = P_ - np.dot(np.linalg.inv(np.dot(J.T, J)), np.dot(J.T, e))[0]

    return P_

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # TODO: Implement this method!
    #Assuming conical cameras K = K' M = [I 0] M'=[R T]
    N_points, M_cameras, _ = image_points.shape
    camera_matrices = np.zeros((M_cameras, 3, 4))
    M1 = np.hstack((K, [[0],[0],[0]]))
    camera_matrices[0] = M1
    RT_S = estimate_initial_RT(E)

    counts = dict()

    for index, RT in enumerate(RT_S):
        M2 = np.dot(K, RT)
        camera_matrices[1] = M2
        for image_point in image_points:
            #P in frame of first cam
            P = nonlinear_estimate_3d_point(image_point, camera_matrices)
            #Transform P to second camera homegeneus coordinates
            P_C2 = np.hstack((P,[1.0]))
            #Apply Transformation
            P_C2 = np.dot(RT, P_C2)
            #Make homogeneus
            P_C2 /= P_C2[-1]
            P_C2 = P_C2[:3]

            if P[2] > 0 and P_C2[2] > 0:
                if index not in counts:
                    counts[index] = 1
                else:
                    counts[index] += 1

    index = sorted(counts.items(), key=lambda x: x[1])[-1][0]

    return RT_S[index]

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = scipy.misc.imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print()
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print()
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c='k', depthshade=True, s=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax.view_init(-100, 90)

    plt.show()
