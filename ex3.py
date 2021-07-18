import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from scipy import signal as sig
from scipy.spatial.distance import cdist

def conv2d(img, kernel):
    h, w = img.shape
    h1, w1 = kernel.shape
    res = np.zeros((h-h1+1, w-w1+1))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = (img[i:i+h1, j:j+w1] * kernel).sum()
    return res

def harris(image, kernel_size, k):
    sobel_para = np.array([[-1, 0, 1]])
    sobel_orth = np.array([[1, 2, 1]])
    sobel_x = np.dot(sobel_orth.T, sobel_para)
    sobel_y = np.dot(sobel_para.T, sobel_orth)

    Ix = sig.convolve2d(image, sobel_x, mode='valid')
    Iy = sig.convolve2d(image, sobel_y, mode='valid')
    # print(Ix[:5, :5])
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    box = np.ones((kernel_size, kernel_size))/kernel_size**2
    sIxx = sig.convolve2d(Ixx, box, mode='valid')
    sIyy = sig.convolve2d(Iyy, box, mode='valid')
    sIxy = sig.convolve2d(Ixy, box, mode='valid')
    # print(Ixx[:5, :5])

    # lambd1 = (sIxx + sIyy + np.sqrt(4 * sIxy + np.square(sIxx - sIyy))) / 2
    # lambd2 = (sIxx + sIyy - np.sqrt(4 * sIxy + np.square(sIxx - sIyy))) / 2
    #
    # scores = lambd1 * lambd2 - k * np.square(lambd1 - lambd2)
    scores = (sIxx * sIyy - sIxy * sIxy) - k * (sIxx + sIyy)**2
    scores[scores<0] = 0
    patch_rad = kernel_size//2
    scores = np.pad(scores, (patch_rad+1, patch_rad+1), 'constant', constant_values=(0, 0))
    return scores

def harris_cv(image, kernel_size, kappa):
    sobel_para = np.array([[-1, 0, 1]])
    sobel_orth = np.array([[1, 2, 1]])
    sobel_x = np.dot(sobel_orth.T, sobel_para)
    sobel_y = np.dot(sobel_para.T, sobel_orth)
    # gaussian_kernel = cv2.getGaussianKernel(3, 0.5)
    gaussian_kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2

    i_x = cv2.filter2D(image, -1, sobel_x, borderType=cv2.BORDER_CONSTANT)
    i_y = cv2.filter2D(image, -1, sobel_y, borderType=cv2.BORDER_CONSTANT)
    # print(i_x[:5, :5])
    sum_i_x_2 = cv2.filter2D(i_x ** 2, -1, gaussian_kernel, borderType=cv2.BORDER_CONSTANT)
    sum_i_y_2 = cv2.filter2D(i_y ** 2, -1, gaussian_kernel, borderType=cv2.BORDER_CONSTANT)
    sum_i_x_y = cv2.filter2D(i_x * i_y, -1, gaussian_kernel, borderType=cv2.BORDER_CONSTANT)
    # print(sum_i_x_2[:5, :5])
    # print(np.average(sum_i_y_2), np.average(sum_i_x_y), np.average(sum_i_x_2))
    result = sum_i_x_2 * sum_i_y_2 - sum_i_x_y ** 2 - kappa * ((sum_i_x_2 + sum_i_y_2) ** 2)
    return result


def shi_tomasi(image, patch_size):
    sobel_para = np.array([[-1, 0, 1]])
    sobel_orth = np.array([[1, 2, 1]])
    sobel_x = np.dot(sobel_orth.T, sobel_para)
    sobel_y = np.dot(sobel_para.T, sobel_orth)

    Ix = sig.convolve2d(image, sobel_x, mode='valid')
    Iy = sig.convolve2d(image, sobel_y, mode='valid')
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    patch = np.ones((patch_size, patch_size))/patch_size**2
    sIxx = sig.convolve2d(Ixx, patch, mode='valid')
    sIyy = sig.convolve2d(Iyy, patch, mode='valid')
    sIxy = sig.convolve2d(Ixy, patch, mode='valid')

    # lambd1 = (sIxx + sIyy + np.sqrt(4 * sIxy + np.square(sIxx - sIyy))) / 2
    # lambd2 = (sIxx + sIyy - np.sqrt(4 * sIxy + np.square(sIxx - sIyy))) / 2
    # scores = np.min(lambd1, lambd2)

    trace = sIxx - sIyy
    det = sIxx * sIyy - sIxy * sIxy
    scores = trace/2 - np.sqrt(np.square(trace/2) - det)
    scores[scores < 0] = 0
    patch_rad = patch_size//2
    scores = np.pad(scores, (patch_rad + 1, patch_rad + 1), 'constant', constant_values=(0, 0))
    return scores

def select_keypoints(arr, k, r):
    indices = []
    h, w = arr.shape
    for _ in range(k):
        ind = np.argmax(arr)
        row = ind // w
        col = ind % w
        indices.append([row, col])
        arr[max(0, row-r):min(h-1, row+r), max(0, col-r):min(w-1, col+r)] = 0
    return np.array(indices)

def describe_keypoints(image, keypoints, r):
    padding_img = np.pad(image, (r, r), 'constant', constant_values=0)
    N = keypoints.shape[0]
    descriptors = np.zeros(((2*r+1)**2, N))
    for i in range(N):
        p = keypoints[i] + r
        sub_img = padding_img[p[0]-r:p[0]+r+1, p[1]-r:p[1]+r+1]
        descriptors[:,i] = np.reshape(sub_img, (-1,))
    return descriptors

def match_descriptors(query_des, database_des, lambd):
    distances = cdist(query_des.T, database_des.T)
    min_dis = np.min(distances, axis=-1)
    min_dis_ind = np.argmin(distances, axis=-1)

    sorted_min_dis = np.sort(min_dis[np.abs(min_dis) > 1e-5])
    min_non_zero_dis = sorted_min_dis[1]
    min_dis_ind[min_dis > lambd * min_non_zero_dis] = 0

    unique_ind = np.zeros(min_dis_ind.shape, dtype=np.int)
    _, w = np.unique(min_dis_ind, return_index=True)
    unique_ind[w] = min_dis_ind[w]
    return unique_ind


if __name__ == '__main__':
    corner_patch_size = 9
    harris_kappa = 0.08
    num_keypoints = 200
    nonmaximum_supression_radius = 8
    descriptor_radius = 9
    match_lambda = 4

    image = cv2.imread("..//Exercise 3 - Simple Keypoint Tracker//data//000000.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    harris_img = harris(image, corner_patch_size, harris_kappa)
    harris_cv(image, corner_patch_size, harris_kappa)
    # print(np.max(harris_img))
    # shi_tomasi_img = shi_tomasi(image, corner_patch_size)

    # cv2.imshow("image", image)
    # cv2.imshow("harris", harris_img)
    # cv2.imshow("shi_tomasi", shi_tomasi_img)
    # cv2.waitKey(0)

    # part 2
    key_points = select_keypoints(harris_img, num_keypoints, nonmaximum_supression_radius)

    # part 3
    descriptors = describe_keypoints(image, key_points, descriptor_radius)
    # for i in range(16):
    #     plt.axis('off')
    #     plt.subplot(4, 4, i+1)
    #     plt.imshow(descriptors[:, i].reshape((19, 19)), cmap=plt.get_cmap('gray'))
    # plt.show()

    image2 = cv2.imread("..//Exercise 3 - Simple Keypoint Tracker//data//000001.png")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    harris_img2 = harris(image2, corner_patch_size, harris_kappa)
    key_points2 = select_keypoints(harris_img2, num_keypoints, nonmaximum_supression_radius)
    descriptors2 = describe_keypoints(image2, key_points2, descriptor_radius)
    # for i in range(16):
    #     plt.axis('off')
    #     plt.subplot(4, 4, i+1)
    #     plt.imshow(descriptors[:, i].reshape((19, 19)), cmap=plt.get_cmap('gray'))
    # plt.show()

    # part 4
    matches = match_descriptors(descriptors2, descriptors, match_lambda)
    plt.figure(figsize=(15, 5))
    plt.axis('off')
    plt.imshow(image2, cmap=plt.get_cmap('gray'))
    plt.scatter(key_points2[:,1], key_points2[:, 0], marker='x', color='r')
    plt.plot(
        [key_points2[np.where(matches != 0), 1].squeeze(),
         key_points[np.array([matches[np.where(matches != 0)]]).astype('int'), 1].squeeze()],
        [key_points2[np.where(matches != 0), 0].squeeze(),
         key_points[np.array([matches[np.where(matches != 0)]]).astype('int'), 0].squeeze()],
        linewidth=4, color='blue'
    )
    plt.show()
    exit()

    # part 5
    img_dir = "..//Exercise 3 - Simple Keypoint Tracker//data//"
    n_img = len(os.listdir(img_dir))
    for i in range(n_img):
        img_name = os.path.join(img_dir, '%06d.png'%i)
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        harris_scores = harris(img, corner_patch_size, harris_kappa)
        cur_keypoints = select_keypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
        cur_des = describe_keypoints(img, cur_keypoints, descriptor_radius)
        # print(cur_keypoints)
        # cur_keypoints = cur_keypoints.T
        if i != 0:
            matches = match_descriptors(cur_des, prev_des, match_lambda)
            plt.figure(figsize=(15, 5))
            plt.axis('off')
            plt.imshow(img, cmap=plt.get_cmap('gray'))
            plt.scatter(cur_keypoints[:, 1], cur_keypoints[:, 0], marker='x', color='r')
            plt.plot(
                [cur_keypoints[np.where(matches != 0), 1].squeeze(),
                 prev_keypoints[np.array([matches[np.where(matches != 0)]]).astype('int'), 1].squeeze()],
                [cur_keypoints[np.where(matches != 0), 0].squeeze(),
                 prev_keypoints[np.array([matches[np.where(matches != 0)]]).astype('int'), 0].squeeze()],
                linewidth=4, color='blue'
            )
            plt.show()
            plt.close()
        prev_des = cur_des
        prev_keypoints = cur_keypoints
