import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cdist
from ex2 import estimate_pose_DLT, reproject_points
from ex3 import harris, select_keypoints, describe_keypoints, match_descriptors
from p3p import P3P
from utils import *

np.random.seed(2) # set the random seed for RANSAC


# def match_descriptors(query_descriptors, database_descriptors, match_lambda):
#     lamda = match_lambda
#
#     dists = cdist(database_descriptors.T, query_descriptors.T, metric='euclidean').transpose()
#     matches = np.argmin(dists, axis=1)
#     dists = np.min(dists, axis=1)
#
#     sorted_dists = np.sort(dists)
#     sorted_dists = sorted_dists[sorted_dists > 0]
#
#     min_non_zero_dist = sorted_dists[0]
#
#     matches[dists >= lamda * min_non_zero_dist] = 0
#
#     # remove double matches
#     _, unique_match_idxs = np.unique(matches, return_index=True)
#
#     unique_matches = np.zeros(matches.shape)
#     unique_matches[unique_match_idxs] = matches[unique_match_idxs]
#
#     return unique_matches

def parabola_ransac(num_iterations, data, max_noise, rerun_on_inliers=True):
    """
    Inputs:
    data - 2 x N matrix with the data points given column-wise

    best_guess_history - 3 x num_iterations ith the polynomial coefficients from polyfit of the BEST GUESS SO FAR at each iteration columnwise

    max_num_inliers_history - 1 x num_iterations with the inlier count of the BEST GUESS SO FAR at each iteration
    """

    best_guess_history = np.zeros((3, num_iterations))
    max_num_inliers_history = np.zeros((1, num_iterations))

    best_guess = np.zeros((3, 1))  # coefficients of polynomial fit to data
    max_num_inliers = 0
    num_samples = 3

    for i in range(num_iterations):
        # Model based on 3 samples
        samples = data[:, np.random.choice(data.shape[1], num_samples, replace=False)]
        guess = np.polyfit(samples[0, :], samples[1, :], 2)  # fit a polynomial to the random sample of x-y points

        # Evaluate number of inliers
        errors = np.abs(np.polyval(guess, data[0, :]) - data[1, :])
        inliers = errors <= max_noise + 1e-5
        num_inliers = np.count_nonzero(inliers)

        # Determine if the current guess is the best so far
        if num_inliers > max_num_inliers:
            if rerun_on_inliers:
                guess = np.polyfit(data[0, inliers], data[1, inliers], 2)

            best_guess = np.transpose(guess)
            max_num_inliers = num_inliers

        best_guess_history[:, i] = best_guess
        max_num_inliers_history[0, i] = max_num_inliers

    return best_guess_history, max_num_inliers_history


def ransac_localization(matched_query_keypoints, corresponding_landmarks, K, use_p3p=False, tweaked_for_more=True,
                        adaptive=True):
    """
    Inputs:
    matched_query_keypoints - keypoints should be 2 x num_keypoints. All matches should be 1 x num_keypoints and correspond to the output from the match_descriptors method of Harris class
    corresponding_landmarks - matched 3D landmarks
    K - camera matrix
    use_p3p
    tweaked_for_more
    adaptive - whether or not to use RANSAC adaptively

    Outputs:
    R_C_W - rotation matrix from world frame to camera
    t_C_W - translation from world frame to camera
    best_inlier_mask
    max_num_inliers_history
    num_iteration_history

    Notes:
    best_inlier_mask should be 1 x num_matched (!!!) and contain, only for the matched keypoints (!!!), 0 if the match is an outlier; 1 otherwise
    """

    # 1. Find keypoints in query image
    if use_p3p:
        if tweaked_for_more:
            num_iterations = 1000
        else:
            num_iterations = 200
        pixel_tolerance = 10
        k = 3  # for P3P method
    else:
        num_iterations = 2000
        pixel_tolerance = 10
        k = 6

    if adaptive:
        num_iterations = np.inf

    # Initialize RANSAC
    best_inlier_mask = np.zeros((1, matched_query_keypoints.shape[0]))

    # (row, col) to (u, v)
    matched_query_keypoints = np.fliplr(matched_query_keypoints)

    max_num_inliers_history = []
    num_iteration_history = []
    max_num_inliers = 0

    # RANSAC
    i = 1
    p3p = P3P()
    K_inv = np.linalg.inv(K)

    M_C_W_guess = np.zeros((3, 4, 4))
    while num_iterations > i:
        # Model from k samples (DLT or P3P)
        idx = np.random.choice(corresponding_landmarks.shape[0], k, replace=False)
        landmark_sample = corresponding_landmarks[idx, :]
        keypoint_sample = matched_query_keypoints[idx, :]

        if use_p3p:
            # Backproject keypoints to unit bearing vectors

            normalized_bearings = np.matmul(K_inv,
                                            np.transpose(np.concatenate((keypoint_sample, np.ones((3, 1))), axis=1)))
            for ii in range(3):
                normalized_bearings[:, ii] /= np.linalg.norm(normalized_bearings[:, ii])

            poses = p3p.p3p(np.transpose(landmark_sample), normalized_bearings)

            # Decode p3p output
            R_C_W_guess = np.zeros((3, 3, 4))
            t_C_W_guess = np.zeros((3, 1, 4))
            for ii in range(4):
                R_W_C_ii = np.real(poses[:, (1 + ii * 4):(4 + ii * 4)])
                t_W_C_ii = np.real(poses[:, 4 * ii])

                R_C_W_guess[:, :, ii] = np.transpose(R_W_C_ii)
                t_C_W_guess[:, :, ii] = -np.matmul(np.transpose(R_W_C_ii), t_W_C_ii[:, np.newaxis])

            M_C_W_guess[:, :3, :] = R_C_W_guess
            M_C_W_guess[:, -1, :] = t_C_W_guess[:, 0, :]

        else:
            M_C_W_guess[:, :, 0] = estimate_pose_DLT(keypoint_sample,
                                                            landmark_sample, K)  # first argument is the 2D correspondence point; second argument is the 3D correspondence point
        #     print(M_C_W_guess[:,:,0])
        # print(keypoint_sample)
        projected_points = reproject_points(landmark_sample, M_C_W_guess[:, :, 0], K)  # p_landmarks,C = R_C_W * P_landmarks,W + t_C_W

        difference = keypoint_sample - projected_points
        # print(projected_points)
        # print(landmark_sample)
        # exit()
        errors = np.sum(np.square(difference), axis=1)
        is_inlier = errors < pixel_tolerance ** 2

        # If we use p3p, also consider inliers for the alternative solutions
        if use_p3p:
            for alt_idx in range(3):
                projected_points = reproject_points(corresponding_landmarks, M_C_W_guess[:, :, 0], K)

                difference = matched_query_keypoints - projected_points
                errors = np.sum(np.square(difference), axis=1)
                alternative_is_inlier = errors < pixel_tolerance ** 2

                if np.count_nonzero(alternative_is_inlier) > np.count_nonzero(is_inlier):
                    is_inlier = alternative_is_inlier

        if tweaked_for_more:
            min_inlier_count = 30
        else:
            min_inlier_count = 6

        # Compute the number of inliers; update the inlier mask
        if np.count_nonzero(is_inlier) > max_num_inliers and np.count_nonzero(is_inlier) >= min_inlier_count:
            max_num_inliers = np.count_nonzero(is_inlier)

            best_inlier_mask = is_inlier

        if adaptive:
            # estimate of the outlier ratio
            outlier_ratio = 1 - max_num_inliers / is_inlier.shape[0]

            # formula to compute number of iterations from estimated outliers
            # ratio
            confidence = 0.95
            upper_bound_on_outlier_ratio = 0.90
            outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio)
            num_iterations = np.log10(1 - confidence) / np.log10(1 - (1 - outlier_ratio) ** k)

            # cap the number of iterations at 15000
            num_iterations = min(15000, num_iterations)

        num_iteration_history.append(num_iterations)
        max_num_inliers_history.append(max_num_inliers)

        i += 1

    if max_num_inliers == 0:
        R_C_W = np.eye(3)
        t_C_W = np.zeros((3, 1))

    else:
        M_C_W = estimate_pose_DLT(matched_query_keypoints[best_inlier_mask > 0, :],
                                         corresponding_landmarks[best_inlier_mask > 0, :], K)
        R_C_W = M_C_W[:, :3]
        t_C_W = M_C_W[:, -1]

    if adaptive:
        print("     Adaptive RANSAC: Needed {} iterations to converge.".format(str(i - 1)))
        print("     Adaptive RANSAC: Estimated outliers: {}%".format(str(int(100 * outlier_ratio))))

    # inlier_mask = best_inlier_mask

    return R_C_W, t_C_W, best_inlier_mask, np.array(max_num_inliers_history), np.array(num_iteration_history)

def plot_matches(matches, query_keypoints, database_keypoints, ax=plt.gca()):

    query_indices = np.argwhere(matches > 0)
    match_indices = matches[matches > 0].astype(int)

    x_from = np.reshape(query_keypoints[query_indices, 0], (-1,1))
    x_to = np.reshape(database_keypoints[match_indices, 0], (-1,1))
    y_from = np.reshape(query_keypoints[query_indices, 1], (-1,1))
    y_to = np.reshape(database_keypoints[match_indices, 1], (-1,1))

    for i in range(y_from.shape[0]):
        ax.plot(np.array([y_from[i], y_to[i]]), np.array([x_from[i], x_to[i]]), color='g', linestyle='-', linewidth=2)

    return

if __name__ == '__main__':

    ## Create data for parts 1 and 2
    num_inliers = 20
    num_outliers = 10
    noise_ratio = 0.1
    poly = np.random.rand(3, 1)  # random second-order polynomial
    extremum = -poly[1] / (2 * poly[0])
    xstart = extremum - 0.5
    lowest = np.polyval(poly, extremum)
    highest = np.polyval(poly, xstart)
    xspan = 1
    yspan = highest - lowest
    max_noise = noise_ratio * yspan
    x = np.random.rand(1, num_inliers) + xstart
    y = np.polyval(poly, x)
    y += (np.random.rand(y.shape[0], y.shape[1]) - 0.5) * 2 * max_noise
    data = np.concatenate((
        np.concatenate((x, np.random.rand(1, num_outliers) + xstart), axis=1),
        np.concatenate((y, np.random.rand(1, num_outliers) * yspan + lowest), axis=1)), axis=0)

    # Part 1
    best_guess_history, max_num_inliers_history = parabola_ransac(500, data, max_noise)

    # Compare with full data fit
    full_fit = np.polyfit(data[0, :], data[1, :], 2)

    _, ax = plt.subplots(1, 2)
    ax[0].scatter(data[0, :], data[1, :], color='b')

    x = np.arange(start=xstart, stop=(xstart + 1.), step=0.01)
    for i in range(best_guess_history.shape[1]):
        guess_plot, = ax[0].plot(x, np.polyval(best_guess_history[:, i], x), color='b')

    truth_plot, = ax[0].plot(x, np.polyval(poly, x), color='g', linewidth=2)

    best_plot, = ax[0].plot(x, np.polyval(best_guess_history[:, -1], x), color='r', linewidth=2)

    fit_plot, = ax[0].plot(x, np.polyval(full_fit, x), color='r', marker=".")

    ax[0].set_xlim(xstart, xstart + 1)
    ax[0].set_ylim(lowest - max_noise, highest + max_noise)

    ax[0].legend((truth_plot, best_plot, fit_plot, guess_plot),
                 ('ground truth', 'RANSAC result', 'full data fit', 'RANSAC guesses'))

    ax[0].set_title('RANSAC vs full fit')

    ax[1].plot(np.arange(0, max_num_inliers_history.shape[1], 1), max_num_inliers_history.squeeze(axis=0))

    ax[1].set_title('Max num inliers vs iterations')

    plt.show(block=False)
    plt.pause(1)
    plt.close()

    # Part 2
    corner_patch_size = 9
    harris_kappa = 0.08
    num_keypoints = 1000
    nonmaximum_supression_radius = 8
    descriptor_radius = 9
    match_lambda = 5

    K = read_K_from_txt("../Exercise 7 - From images to localization/data/K.txt")
    p_W_landmarks = read_P_w_from_txt("../Exercise 7 - From images to localization/data/p_W_landmarks.txt")

    database_image = cv2.imread("../Exercise 7 - From images to localization/data/000000.png", cv2.IMREAD_GRAYSCALE)
    # database_harris = harris(database_image, corner_patch_size, harris_kappa)
    # database_keypoints = select_keypoints(database_harris, num_keypoints, nonmaximum_supression_radius)
    database_keypoints = read_keypoints_from_txt("../Exercise 7 - From images to localization/data/keypoints.txt")
    database_descriptors = describe_keypoints(database_image, database_keypoints, descriptor_radius)


    query_image = cv2.imread("../Exercise 7 - From images to localization/data/000001.png", cv2.IMREAD_GRAYSCALE)
    query_harris = harris(query_image, corner_patch_size, harris_kappa)
    query_keypoints = select_keypoints(query_harris, num_keypoints, nonmaximum_supression_radius)
    query_descriptors = describe_keypoints(query_image, query_keypoints, descriptor_radius)

    matches = match_descriptors(query_descriptors, database_descriptors, match_lambda)

    # plt.figure(figsize=(15, 5))
    # plt.axis('off')
    # plt.imshow(query_image, cmap=plt.get_cmap('gray'))
    # plt.scatter(query_keypoints[:, 1], query_keypoints[:, 0], marker='x', color='r')
    # plt.plot(
    #     [query_keypoints[np.where(matches != 0), 1].squeeze(),
    #      database_keypoints[np.array([matches[np.where(matches != 0)]]).astype('int'), 1].squeeze()],
    #     [query_keypoints[np.where(matches != 0), 0].squeeze(),
    #      database_keypoints[np.array([matches[np.where(matches != 0)]]).astype('int'), 0].squeeze()],
    #     linewidth=4, color='blue'
    # )
    # plt.show()
    # print(matches)

    matched_query_keypoints = query_keypoints[np.where(matches != 0), :].squeeze()

    corresponding_matches = matches[matches != 0].astype(np.int)
    corresponding_landmarks = p_W_landmarks[corresponding_matches, :]

    matched_database_keypoints = database_keypoints[corresponding_matches, :]

    R_C_W, t_C_W, inlier_mask, max_num_inliers_history, num_iteration_history = ransac_localization(
        matched_query_keypoints, corresponding_landmarks, K, use_p3p=True, tweaked_for_more=True, adaptive=True)

    print(R_C_W, t_C_W)


    _, ax = plt.subplots(3, 1)

    ax[0].imshow(query_image)
    # ax[0].plot(query_keypoints[:, 1], query_keypoints[:, 0], c='r', marker='x', linewidth=1,
    #            linestyle=" ")
    ax[0].scatter(query_keypoints[:, 1], query_keypoints[:, 0], marker='x', color='r')
    ax[0].plot(
        [query_keypoints[np.where(matches != 0), 1].squeeze(),
         database_keypoints[np.array([matches[np.where(matches != 0)]]).astype('int'), 1].squeeze()],
        [query_keypoints[np.where(matches != 0), 0].squeeze(),
         database_keypoints[np.array([matches[np.where(matches != 0)]]).astype('int'), 0].squeeze()],
        linewidth=1, color='blue'
    )
    ax[0].set_title("All keypoints and matches")

    ax[1].imshow(query_image)
    ax[1].plot(matched_query_keypoints[(1 - inlier_mask) > 0, 1],
               matched_query_keypoints[(1 - inlier_mask) > 0, 0], c='r', marker='x', linewidth=1, linestyle=" ")
    ax[1].plot(matched_query_keypoints[inlier_mask > 0, 1], matched_query_keypoints[inlier_mask > 0, 0],
               c='g', marker='x', linestyle=" ")
    # ax[1].scatter(query_keypoints[:, 1], query_keypoints[:, 0], marker='x', color='r')
    ax[1].plot(
        [matched_query_keypoints[inlier_mask > 0, 1].squeeze(),
         matched_database_keypoints[inlier_mask > 0, 1].squeeze()],
        [matched_query_keypoints[inlier_mask > 0, 0].squeeze(),
         matched_database_keypoints[inlier_mask > 0, 0].squeeze()],
        linewidth=1, color='blue'
    )
    ax[1].set_title("Inlier and outlier matches")

    ax[2].plot(np.arange(1, max_num_inliers_history.shape[0] + 1), max_num_inliers_history)
    ax[2].set_title("Maximum inlier count over RANSAC iterations")

    plt.show(block=False)
    plt.pause(1)
    plt.close()

