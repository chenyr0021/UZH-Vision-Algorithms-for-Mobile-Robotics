import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
import open3d as o3d

def get_disparity(left_img, right_img, patch_rad, min_disp, max_disp):
    h, w = left_img.shape
    disp = np.zeros((h, w), dtype=np.uint8)
    for i in range(patch_rad, h - patch_rad):
        for j in range(patch_rad + max_disp, w - patch_rad):
            ssd = []
            left_patch = left_img[i-patch_rad:i+patch_rad+1, j-patch_rad:j+patch_rad+1].astype(np.float)
            for d in range(max_disp, min_disp-1, -1):
                if j-d-patch_radius >= 0:
                    right_patch = right_img[i-patch_rad:i+patch_rad+1, j-d-patch_rad:j-d+patch_rad+1].astype(np.float)
                    ssd.append(np.sum(np.square(left_patch - right_patch)))
            min_ssd = np.min(ssd)
            neg_disp = np.argmin(ssd)
            if np.sum(ssd < 1.5 * min_ssd) < 3 and neg_disp != 0 and neg_disp != len(ssd) - 1:
                disp[i, j] = max_disp - neg_disp

            # plt.plot(disp_candidates)
            # plt.show()
            # exit()
            # print(opt_disp)
        print(i)
    return disp

def disparity_to_pointcloud(disp_img, K, baseline, left_img):
    h, w = left_img.shape
    p_left = np.asarray([[i, j, 1] for i in range(h) for j in range(w)])
    disp_v = np.resize(disp_img, (h*w,))
    p_right = p_left
    p_right[:, 1] += disp_v

    p_left = p_left[disp_v > 0, :]
    p_right = p_right[disp_v > 0, :]

    bv_left = np.matmul(np.linalg.inv(K), p_left.T)
    bv_right = np.matmul(np.linalg.inv(K), p_right.T)

    points = np.zeros(bv_left.shape)
    b = np.array([[baseline], [0], [0]])

    for i in range(points.shape[1]):
        A = np.concatenate([bv_left[:, i:i+1], bv_right[:, i:i+1]], axis=1)
        lambd = np.matmul(np.linalg.pinv(np.matmul(A.T, A)), np.matmul(A.T, b))
        points[:, i] = bv_left[:, i] * lambd[0]

    return points


if __name__ == '__main__':
    baseline = 0.54
    patch_radius = 5
    min_disp = 5
    max_disp = 50
    xlims = [7, 20]
    ylims = [-6, 10]
    zlims = [-5, 5]

    left_img = cv2.imread("E:\chenyr\Desktop\VSLAM\Exercise 5 - Stereo Dense Reconstruction\data\left\\000000.png", cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread("E:\chenyr\Desktop\VSLAM\Exercise 5 - Stereo Dense Reconstruction\data\\right\\000000.png", cv2.IMREAD_GRAYSCALE)
    h, w = left_img.shape
    scale = 5
    left_img = cv2.resize(left_img, (w//scale, h//scale))
    right_img = cv2.resize(right_img, (w//scale, h//scale))
    K = utils.read_K_from_txt("E:\chenyr\Desktop\VSLAM\Exercise 5 - Stereo Dense Reconstruction\data\K.txt")
    print(left_img.shape)
    # K[:2, :] = K[:2, :] / 2

    disp = get_disparity(left_img, right_img, patch_radius, min_disp, max_disp)

    # for i in range(disp.shape[0]):
    #     print(disp[i])

    # cv2.imshow("disp", disp * 5)
    # cv2.waitKey(0)

    P_c = disparity_to_pointcloud(disp, K, baseline, left_img)
    P_w = np.linalg.inv(np.mat([[0, -1, 0], [0, 0, -1], [1, 0, 0]])) * P_c[:, 0:-1:10]
    print(P_w)
    cloud = o3d.geometry.PointCloud(P_w)
    # cloud.points = P_w.T

