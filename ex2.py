import numpy as np
import cv2
import utils
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_Q(p_wave, P_w):
    p_wave = np.asarray(p_wave)
    P_w = np.asarray(P_w)
    rows = p_wave.shape[0]
    Q = np.zeros((2 * rows, 12))
    Q[::2, [0, 1, 2, 3]] = np.concatenate([P_w, np.ones((rows, 1))], axis=1)
    Q[::2, [8, 9, 10, 11]] = np.concatenate([P_w, np.ones((rows, 1))], axis=1) * p_wave[:, 0].reshape(-1, 1) * (-1)
    Q[1::2, [4, 5, 6, 7]] = np.concatenate([P_w, np.ones((rows, 1))], axis=1)
    Q[1::2, [8, 9, 10, 11]] = np.concatenate([P_w, np.ones((rows, 1))], axis=1) * p_wave[:, 1].reshape(-1, 1) * (-1)
    return Q

def estimate_pose_DLT(p, P_w, K):
    P_w = np.asarray(P_w)
    p = np.asarray(p)
    p_wave = np.dot(np.linalg.inv(K), np.concatenate([p, np.ones((p.shape[0], 1))], axis=1).T).T
    Q = generate_Q(p_wave, P_w)
    U, S, V_T = np.linalg.svd(Q)
    M_tilde = V_T[-1, :]
    M_tilde = M_tilde.reshape((3, 4))
    if np.linalg.det(M_tilde[:, :3]) < 0:
        M_tilde = M_tilde * (-1)
    R = M_tilde[:, :3]
    U, S, V_T = np.linalg.svd(R)
    R_tilde = np.dot(U, V_T)
    alpha = np.linalg.norm(R_tilde) / np.linalg.norm(R)
    return np.concatenate([R_tilde, alpha * M_tilde[:,3].reshape((3, 1))], axis=1)

def reproject_points(P_w, M, K):
    P_w = np.asarray(P_w)
    p = np.dot(K, np.dot(M, np.concatenate([P_w, np.ones((P_w.shape[0], 1))], axis=1).T)).T
    p[:, 0] /= p[:, 2]
    p[:, 1] /= p[:, 2]
    p = np.rint(p[:, :2]).astype('int')
    return p


def inverse_M(M):
    R = M[:, :3]
    t = M[:, 3:]
    R = np.linalg.inv(R)
    t = -t
    return np.concatenate([R, t], axis=1)


if __name__ == '__main__':
    root = "C:\\Users\\narwal.DESKTOP-932GS5J"
    # root = "E:\chenyr"
    K = utils.read_K_from_txt(root + "\Desktop\VSLAM\Exercise 2 - PnP\data\K.txt")
    P_w = utils.read_P_w_from_txt(root + "\Desktop\VSLAM\Exercise 2 - PnP\data\p_W_corners.txt")
    p = utils.read_p_wave_from_txt(root + "\Desktop\VSLAM\Exercise 2 - PnP\data\detected_corners.txt")
    image = cv2.imread(root + "\Desktop\VSLAM\Exercise 2 - PnP\data\images_undistorted\img_0001.jpg")
    # 1
    M = estimate_pose_DLT(p[0, :, :], P_w, K)
    for u, v in p[0, :, :].astype('int'):
        cv2.circle(image, (u, v), radius=3, color=(0, 0, 255), thickness=1)

    # 2
    # reproj = reproject_points(P_w, M, K)
    # for u, v in reproj:
    #     cv2.circle(image, (u, v), radius=3, color=(0, 255, 0), thickness=1)
    #
    # cv2.imshow("reprojection", image)
    # cv2.waitKey(0)

    # 3
    M_inv = inverse_M(M)
    ends = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]) * 0.02
    img_dir = root + "\Desktop\VSLAM\Exercise 2 - PnP\data\images_undistorted"
    img_num = len(os.listdir(img_dir))
    xs = []
    ys = []
    zs = []
    for i in range(img_num):
        img = cv2.imread(os.path.join(img_dir, "img_%04d.jpg" % (i+1)))
        Mi = estimate_pose_DLT(p[i, :, :], P_w, K)
        Mi_inv = inverse_M(Mi)
        ends_ = np.dot(Mi_inv, np.concatenate([ends, np.ones((ends.shape[0], 1))], axis=1).T).T
        # ???????????????
        fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        x = ends_[:, 0]
        y = ends_[:, 1]
        z = ends_[:, 2]
        ax1.set_xlim(-0.1, 0.3)
        ax1.set_ylim(0, 0.2)
        ax1.set_zlim(-0.5, -0.2)
        # ax1.scatter3D(x, y, z, cmap='blues')
        ax1.plot3D(x[[0, 1]], y[[0, 1]], z[[0, 1]])
        ax1.plot3D(x[[0, 2]], y[[0, 2]], z[[0, 2]])
        ax1.plot3D(x[[0, 3]], y[[0, 3]], z[[0, 3]])

        # x = ends[:, 0]
        # y = ends[:, 1]
        # z = ends[:, 2]
        #
        # # ax1.scatter3D(x, y, z, cmap='blues')
        # ax1.plot3D(x[[0, 1]], y[[0, 1]], z[[0, 1]])
        # ax1.plot3D(x[[0, 2]], y[[0, 2]], z[[0, 2]])
        # ax1.plot3D(x[[0, 3]], y[[0, 3]], z[[0, 3]])
        fig.savefig(root + "\Desktop\VSLAM\Exercise 2 - PnP" + "\\fig_%04d.png"%(i+1))
        print(i)
        # plt.show()
        plt.close()
        xs.append(ends[0, 0])
        ys.append(ends[0, 1])
        zs.append(ends[0, 2])
        # exit()
    # print(np.min(xs))
    # print(np.min(ys))
    # print(np.min(zs))
    # print(np.max(xs))
    # print(np.max(ys))
    # print(np.max(zs))

    video_name = 'cam_motion.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_name,fourcc=fourcc, fps=30.0, frameSize=(640, 480), isColor=True)
    for i in range(210):
        image = cv2.imread(root + "\Desktop\VSLAM\Exercise 2 - PnP" + "\\fig_%04d.png"%(i+1))
        video_writer.write(image)
        print("write frame: ", i)
        cv2.imshow(" ", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    video_writer.release()
