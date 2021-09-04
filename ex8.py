import numpy as np
import cv2#.cv2 as cv2
import matplotlib.pyplot as plt
import utils

def get_sim_warp(dx, dy, alpha, lambd):
    alpha_rad = alpha * np.pi / 180
    return lambd * np.array([[np.cos(alpha_rad), -np.sin(alpha_rad), dx],
                             [np.sin(alpha_rad), np.cos(alpha_rad), dy]])

def warp_image(image, W):
    h, w = image.shape
    res = np.zeros(image.shape)
    for i in range(w):
        for j in range(h):
            warped = np.dot(W, np.array([[i, j, 1]]).T).T
            w_w, h_w = warped[0, 0], warped[0, 1]
            if 0 <= h_w < h and 0 <= w_w < w:
                res[j][i] = image[int(h_w), int(w_w)]
    return res

def get_warped_patch(image, W, x_T, r_T, interpolate=True):
    h, w = image.shape
    patch = np.zeros((2*r_T+1, 2*r_T+1))
    for i in range(-r_T, r_T+1):
        for j in range(-r_T, r_T+1):
            warped = x_T + np.dot(W, np.array([[i, j, 1]]).T).T
            w_w, h_w = warped[0, 0], warped[0, 1]
            warped = warped.squeeze()
            if 0 <= h_w < h-1 and 0 <= w_w < w-1:
                if interpolate:
                    floors = np.floor(warped)
                    a, b = (warped - floors).tolist()
                    intensity = (1 - b) * ((1 - a) * image[int(floors[1]), int(floors[0])]
                                + a * image[int(floors[1]), int(floors[0]) + 1]) \
                                + b * ((1 - a) * image[int(floors[1]) + 1, int(floors[0])]
                                + a * image[int(floors[1]) + 1, int(floors[0]) + 1])
                    patch[j + r_T, i + r_T] = intensity
                else:
                    patch[j + r_T, i + r_T] = image[int(h_w), int(w_w)]
    return patch

def track_brute_force(I_R, I, x_T, r_T, r_D):
    ssds = np.zeros((2*r_D+1, 2*r_D+1))
    center_x, center_y = x_T.tolist()
    template = I_R[center_y-r_T:center_y+r_T+1, center_x-r_T:center_x+r_T+1]
    for dx in range(-r_D, r_D+1):
        for dy in range(-r_D, r_D+1):
            patch = get_warped_patch(I, get_sim_warp(dx, dy, 0, 1), x_T, r_T)
            ssd = np.sum(np.square(patch - template))
            ssds[dx + r_D, dy + r_D] = ssd
    dx = np.argmin(ssds) - r_D
    return dx, ssds

def conv2(v1, v2, m, mode='same'):
    tmp = np.apply_along_axis(np.convolve, 0, m, v1, mode)
    return np.apply_along_axis(np.convolve, 1, tmp, v2, mode)
    
def track_KLT(I_R, I, x_T, r_T, num_iters, do_plot=False):
    p_hist = np.zeros((6, num_iters+1))
    W = get_sim_warp(0, 0, 0, 1)
    p_hist[:, 0] = np.reshape(W, (6,))
    
    I_RT = get_warped_patch(I_R, W, x_T, r_T)
    i_R = np.reshape(I_RT, (I_RT.shape[0] * I_RT.shape[1], 1), order='F')
    
    xs = np.arange(-r_T, r_T+1)
    ys = xs.copy()
    n = xs.shape[0]
    xy1 = np.concatenate([np.kron(xs, np.ones((1, n))).T,
                          np.kron(np.ones((1, n)), ys).T,
                          np.ones((n*n, 1))], axis=1)
    dwdp = np.kron(xy1, np.eye(2))
    
    if do_plot:
        fig, ax = plt.subplots(3, 1)
    
    kernel = np.array([1, 0, -1])
    for it in range(num_iters):
        big_IWT = get_warped_patch(I, W, x_T, r_T+1)
        IWT = big_IWT[1:-1, 1:-1]
        i = np.reshape(IWT, (IWT.shape[0]*IWT.shape[1], -1), order='F')
        
        # tmp = np.apply_along_axis(np.convolve, 0, big_IWT[1:-1, :], 1, 'valid')
        # IWTx = np.apply_along_axis(np.convolve, 1, tmp, kernel, 'valid')
        # tmp = np.apply_along_axis(np.convolve, 0, big_IWT[:, 1:-1], kernel, 'valid')
        # IWTy = np.apply_along_axis(np.convolve, 1, tmp, 1, 'valid')
        
        IWTx = conv2(1, kernel, big_IWT[1:-1, :], mode='valid')
        IWTy = conv2(kernel, 1, big_IWT[:, 1:-1], mode='valid')
        
        
        didw = np.concatenate([np.reshape(IWTx, (-1, 1), order='F'), np.reshape(IWTy, (-1, 1), order='F')], axis=1)
        didp = np.zeros((n*n, 6))
        for p_i in range(n*n):
            didp[p_i, :] = np.dot(didw[p_i, :], dwdp[p_i*2:p_i*2+2, :])
        
        H = np.dot(np.transpose(didp), didp)

        delta_p = np.dot(np.dot(np.linalg.inv(H), np.transpose(didp)), i_R - i)
        W += np.reshape(delta_p, (2, 3), order='F')

        if do_plot:
            tmp_mat = np.concatenate((IWT, I_RT, (I_RT - IWT)), axis=1)
            ax[0].imshow(tmp_mat)
            ax[0].set_title('I(W(T)), I_R(T) and the difference')
            ax[0].get_xaxis().set_visible(False)
            ax[0].get_yaxis().set_visible(False)
    
            tmp_mat = np.concatenate((IWTx, IWTy), axis=1)
            ax[1].imshow(tmp_mat)
            ax[1].set_title('warped gradients')
            ax[1].get_xaxis().set_visible(False)
            ax[1].get_yaxis().set_visible(False)
    
            descentcat = np.zeros((n, 6 * n))
            for j in range(6):
                descentcat[:, j * n:(j + 1) * n] = np.reshape(didp[:, j], (n, n))
            ax[2].imshow(descentcat)
            ax[2].set_title('steepest descent images')
            ax[2].get_xaxis().set_visible(False)
            ax[2].get_yaxis().set_visible(False)
    
            plt.pause(0.1)
        
        p_hist[:, it+1] = np.reshape(W, (-1,))
        
        if np.linalg.norm(delta_p) < 1e-3:
            p_hist = p_hist[:, :it+1]
            break
    if do_plot:
        plt.close()
    
    return W, p_hist

def plot_matches(matches, query_keypoints, database_keypoints):

    query_indices = np.squeeze(np.argwhere(matches > -1))
    match_indices = matches[matches > -1].astype(int)

    x_from = np.reshape(query_keypoints[query_indices, 0], (-1, 1))
    x_to = np.reshape(database_keypoints[match_indices, 0], (-1, 1))
    y_from = np.reshape(query_keypoints[query_indices, 1], (-1, 1))
    y_to = np.reshape(database_keypoints[match_indices, 1], (-1, 1))

    for i in range(y_from.shape[0]):
        plt.plot(np.array([y_from[i], y_to[i]]), np.array([x_from[i], x_to[i]]), color='g', linestyle='-',
                linewidth=2)

    return

def track_KLT_robustly(I_prev, I, keypoint, r_T, num_iters, lamba):
    W, _ = track_KLT(I_prev, I, keypoint, r_T, num_iters)
    delta_keypoint = W[:, -1]
    W_inv, _ = track_KLT(I, I_prev, (keypoint.T + delta_keypoint).T, r_T, num_iters, do_plot=False)
    dkp_inv = W_inv[:, -1]
    keep = np.linalg.norm(delta_keypoint + dkp_inv) < lamba
    return delta_keypoint, keep

if __name__ == '__main__':
    # Part 1
    # image = cv2.imread("../Exercise 8 - Lucas-Kanade tracker/data/000000.png", cv2.IMREAD_GRAYSCALE)
    # plt.figure(1)
    # plt.subplot('221')
    # plt.imshow(image)
    # plt.title("Reference")
    #
    # plt.subplot('222')
    # W = get_sim_warp(50, -30, 0, 1)
    # plt.imshow(warp_image(image, W))
    # plt.title("Translation")
    #
    # plt.subplot('223')
    # W = get_sim_warp(0, 0, 10, 1)
    # plt.imshow(warp_image(image, W))
    # plt.title('Rotation around upper left corner')
    #
    # plt.subplot('224')
    # W = get_sim_warp(0, 0, 0, 0.5)
    # plt.imshow(warp_image(image, W))
    # plt.title('Zoom on upper left corner')
    # plt.show()
    
    # Part 2
    # I_R = cv2.imread("../Exercise 8 - Lucas-Kanade tracker/data/000000.png", cv2.IMREAD_GRAYSCALE)
    # plt.figure(2)
    # plt.subplot('121')
    # W0 = get_sim_warp(0, 0, 0, 1)
    # x_T = np.array([900, 291])
    # r_T = 15
    # template = get_warped_patch(I_R, W0, x_T, r_T)
    # plt.imshow(template)
    #
    # plt.subplot('122')
    # W = get_sim_warp(10, 6, 0, 1)
    # I = warp_image(I_R, W)
    # r_D = 20
    # dx, ssds = track_brute_force(I_R, I, x_T, r_T, r_D)
    # plt.imshow(ssds)
    # plt.title("SSDs")
    # plt.show()
    
    # Part 3
    # I_R = cv2.imread("../Exercise 8 - Lucas-Kanade tracker/data/000000.png", cv2.IMREAD_GRAYSCALE)
    # x_T = np.array([900, 291])
    # r_T = 15
    # num_iters = 50
    # W = get_sim_warp(10, 6, 0, 1)
    # I = warp_image(I_R, W)
    # W, p_hist = track_KLT(I_R, I, x_T, r_T, num_iters, do_plot=True)
    # print(W)
    
    # Part 4
    # r_T = 15
    # num_iters = 50
    # data_dir = "../Exercise 8 - Lucas-Kanade tracker/data/"
    # I_R = cv2.imread( data_dir + "000000.png", cv2.IMREAD_GRAYSCALE)
    # I_R = cv2.resize(I_R, dsize=(int(0.25 * I_R.shape[1]), int(0.25 * I_R.shape[0])), interpolation=cv2.INTER_CUBIC)
    # keypoints = np.loadtxt("../Exercise 8 - Lucas-Kanade tracker/data/keypoints.txt", dtype=np.double, comments='#')/4
    # keypoints = np.flipud(np.transpose(keypoints[:50, :]))
    #
    # plt.figure(4)
    # plt.imshow(I_R)
    # plt.scatter(keypoints[0, :], keypoints[1, :], c='r', marker='x')
    # plt.pause(0.1)
    #
    # I_prev = I_R
    #
    # for i in range(1, 21):
    #     I = cv2.imread(data_dir + '%06d.png'%i, cv2.IMREAD_GRAYSCALE)
    #     I = cv2.resize(I, dsize=(int(0.25 * I.shape[1]), int(0.25 * I.shape[0])), interpolation=cv2.INTER_CUBIC)
    #     plt.imshow(I)
    #
    #     dkp = np.zeros(keypoints.shape)
    #     for j in range(keypoints.shape[1]):
    #         W, _ = track_KLT(I_prev, I, keypoints[:, j].T, r_T, num_iters, do_plot=False)
    #         dkp[:, j] = W[:, -1]
    #
    #     kp_old = keypoints.copy()
    #     keypoints = keypoints + dkp
    #     I_prev = np.copy(I)
    #     plt.scatter(kp_old[0, :], kp_old[1, :], c='r', marker='x')
    #
    #     plot_matches(np.arange(0, keypoints.shape[1]), np.flipud(keypoints).transpose(), np.flipud(kp_old).transpose())
    #
    #     plt.pause(0.1)
    # plt.close()
    
    # Part 5
    r_T = 15
    num_iters = 50
    lambd = 0.1
    
    data_dir = "../Exercise 8 - Lucas-Kanade tracker/data/"
    I_R = cv2.imread(data_dir + "000000.png", cv2.IMREAD_GRAYSCALE)
    I_R = cv2.resize(I_R, dsize=(int(0.25 * I_R.shape[1]), int(0.25 * I_R.shape[0])), interpolation=cv2.INTER_CUBIC)
    keypoints = np.loadtxt("../Exercise 8 - Lucas-Kanade tracker/data/keypoints.txt", dtype=np.double, comments='#') / 4
    keypoints = np.flipud(np.transpose(keypoints[:50, :]))

    plt.figure(5)
    plt.imshow(I_R)
    plt.scatter(keypoints[0, :], keypoints[1, :], c='r', marker='x')
    plt.pause(0.1)

    I_prev = I_R

    for i in range(1, 21):
        I = cv2.imread(data_dir + '%06d.png' % i, cv2.IMREAD_GRAYSCALE)
        I = cv2.resize(I, dsize=(int(0.25 * I.shape[1]), int(0.25 * I.shape[0])), interpolation=cv2.INTER_CUBIC)
        plt.imshow(I)
    
        dkp = np.zeros(keypoints.shape)
        keep = np.ones((keypoints.shape[1]), dtype=np.bool)
        for j in range(keypoints.shape[1]):
            dkp[:, j], keep[j] = track_KLT_robustly(I_prev, I, keypoints[:, j].T, r_T, num_iters, lambd)
    
        kp_old = keypoints[:, keep]
        keypoints = keypoints + dkp
        keypoints = keypoints[:, keep]
        I_prev = np.copy(I)
        plt.scatter(kp_old[0, :], kp_old[1, :], c='r', marker='x')
    
        plot_matches(np.arange(0, keypoints.shape[1]), np.flipud(keypoints).transpose(), np.flipud(kp_old).transpose())
    
        plt.pause(0.1)
    plt.close()