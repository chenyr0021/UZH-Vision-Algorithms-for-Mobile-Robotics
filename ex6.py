import numpy as np
import utils
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_points(path):
    P = []
    with open(path, 'r') as f:
        for line in f.readlines():
            nums = line.strip().split(' ')
            nums = [float(i) for i in nums]
            P.append(nums)
    P.append([1] * len(P[0]))
    P = np.asarray(P)
    return P

def cross_to_matrix(x):
    M = [[0, -x[2],  x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]]
    return np.asarray(M)


def linear_triangulation(P1, P2, M1, M2):
    """
    :param P1: 3xn
    :param P2: 3xn
    :param M1: 3x4
    :param M2: 3x4
    :return: P 4xn
    """
    n = P1.shape[1]
    P = np.zeros((4, n))
    for i in range(n):
        p1 = P1[:, i]
        p2 = P2[:, i]
        A = np.concatenate([np.dot(cross_to_matrix(p1), M1), np.dot(cross_to_matrix(p2), M2)], axis=0)
        U, S, V_T = np.linalg.svd(A)
        P[:, i] = V_T[-1, :]
    P = P / P[3, :]
    return P

def fundamental_eight_point(P1, P2):
    """
    :param P1: 3xn
    :param P2: 3xn
    :return: F 3x3
    """
    n  = P1.shape[1]
    Q = np.zeros((n, 9), dtype=np.float)
    for i in range(n):
        Q[i, :] = np.kron(P1[:, i], P2[:, i])
    U, S, Vh = np.linalg.svd(Q)
    F = np.reshape(Vh[-1, :], (3, 3))
    U, S, Vh = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(np.dot(U, np.diag(S)), Vh)
    return F

def fundamental_eight_point_normalized(P1, P2):
    T1, normalized_noisy_P1 = normalize_2d_points(P1)
    T2, normalized_noisy_P2 = normalize_2d_points(P2)
    
    F_tilda = fundamental_eight_point(P1, P2)
    F = np.dot(np.dot(T2.T, F_tilda), T1)
    return F / F[-1, -1]

def dist_point_to_epipolar_line(F, P1, P2):
    n = P1.shape[1]
    
    homog_points = np.concatenate([P1, P2], axis=1)
    epi_lines = np.concatenate([np.dot(F.T, P2), np.dot(F.T, P1)], axis=1)
    denom = epi_lines[0, :] ** 2 + epi_lines[1, :] ** 2
    cost = np.sqrt(np.sum((np.sum(epi_lines * homog_points, axis=0) ** 2) / denom)/ n)
    return cost

def normalize_2d_points(P):
    """
    :param P: 3xn
    :returns:
    new_points 3xn
    T 3x3
    """
    homog_P = P / P[2, :]
    mu = np.zeros((P.shape[0], 1))
    mu[:, 0] = np.mean(homog_P, axis=1)
    # sigma = np.sqrt(np.mean(np.sum(np.square((homog_P - mu)[:2, :]))))
    sigma = np.std(P[:-1, :])
    s = np.sqrt(2) / sigma
    T = np.array([[s, 0, -s * mu[0, 0]],
                  [0, s, -s * mu[1, 0]],
                  [0, 0, 1]])
    P_tilda = np.dot(T, homog_P)
    return T, P_tilda

def estimate_essential_matrix(P1, P2, K1, K2):
    F = fundamental_eight_point_normalized(P1, P2)
    E = np.dot(np.dot(K2.T, F), K1)
    return E

def decompose_essential_matrix(E):
    U, S, V_T = np.linalg.svd(E)
    u = U[:, 2].reshape((3, 1))
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    R1 = np.dot(np.dot(U, W), V_T)
    if np.linalg.det(R1) < 0:
        R1 *= -1
    R2 = np.dot(np.dot(U, W.T), V_T)
    if np.linalg.det(R2) < 0:
        R2 *= -1
    if np.linalg.norm(u) != 0:
        u = u / np.linalg.norm(u)
    return [R1, R2], u

def disambiguate_relative_pose(Rs, u3, P1, P2, K1, K2):
    M1 = np.dot(K1, np.eye(3, 4))
    R_opt = np.eye(3)
    T_opt = np.zeros((3, 1))
    total_points_in_front_best = 0
    for R in Rs:
        for sign in [-1, 1]:
            T = u3 * sign
            Mi = np.dot(K2, np.concatenate([R, T], axis=1))
            PC1 = linear_triangulation(P1, P2, M1, Mi)
            PC2 = np.dot(np.concatenate([R, T], axis=1), PC1)
            
            n_points_in_front1 = np.sum(PC1[2, :] > 0)
            n_points_in_front2 = np.sum(PC2[2, :] > 0)
            total = n_points_in_front1 + n_points_in_front2
            
            if total > total_points_in_front_best:
                R_opt = R
                T_opt = T
                total_points_in_front_best = total
    return np.concatenate([R_opt, T_opt], axis=1)
    

if __name__ == '__main__':
    np.random.seed(0)
    
    
    # Linear Triangulation
    # N = 10
    # P = np.random.randn(4, N)
    # P[3, :] = 1
    #
    # M1 = np.array([[500, 0, 320, 0],
    #                [0, 500, 240, 0],
    #                [0, 0, 1, 0]])
    # M2 = np.array([[500, 0, 320, -100],
    #                [0, 500, 240, 0],
    #                [0, 0, 1, 0]])
    #
    # P1 = np.dot(M1, P)
    # P2 = np.dot(M2, P)
    # P_est = linear_triangulation(P1, P2, M1, M2)
    # print(P - P_est)
    
    # Eight-point algorithm
    # N = 40
    # P = np.random.randn(4, N)
    # P[3, :] = 1
    #
    # M1 = np.array([[500, 0, 320, 0],
    #                [0, 500, 240, 0],
    #                [0, 0, 1, 0]])
    # M2 = np.array([[500, 0, 320, -100],
    #                [0, 500, 240, 0],
    #                [0, 0, 1, 0]])
    #
    # P1 = np.dot(M1, P)
    # P2 = np.dot(M2, P)
    #
    # sigma = 1e-1
    # noisy_P1 = P1 + sigma * np.random.randn(P1.shape[0], P1.shape[1])
    # noisy_P2 = P2 + sigma * np.random.randn(P2.shape[0], P2.shape[1])
    
    # F = fundamental_eight_point(P1, P2)
    # cost_algebraic  = np.linalg.norm(np.sum(P2 * np.dot(F, P1))) / np.sqrt(N)
    # cost_dist_epi_line = dist_point_to_epipolar_line(F, P1, P2)
    # print(cost_algebraic)
    # print(cost_dist_epi_line)
    #
    # F = fundamental_eight_point_normalized(noisy_P1, noisy_P2)
    # cost_algebraic = np.linalg.norm(np.sum(noisy_P2 * np.dot(F, noisy_P1))) / np.sqrt(N)
    # cost_dist_epi_line = dist_point_to_epipolar_line(F, noisy_P1, noisy_P2)
    # print(cost_algebraic)
    # print(cost_dist_epi_line)
    
    # Fn = fundamental_eight_point_normalized(noisy_P1, noisy_P1)
    # cost_algebraic = np.linalg.norm(np.sum(noisy_P2 * np.dot(Fn, noisy_P1))) / np.sqrt(N)
    # cost_dist_epi_line = dist_point_to_epipolar_line(Fn, noisy_P1, noisy_P2)
    # print(cost_algebraic)
    # print(cost_dist_epi_line)
    
    
    # Putting things together: Structure from Motion

    K = np.array([[1379.74, 0, 760.35],
                  [0, 1382.08, 503.41],
                  [0, 0, 1]])
    P1_path = "../Exercise 6 - Two-view Geometry/data/matches0001.txt"
    P2_path = "../Exercise 6 - Two-view Geometry/data/matches0002.txt"
    P1 = load_points(P1_path)
    P2 = load_points(P2_path)

    E = estimate_essential_matrix(P1, P2, K, K)
    
    Rs, u3 = decompose_essential_matrix(E)
    transf = disambiguate_relative_pose(Rs, u3, P1, P2, K, K)
    print(transf)
    
    M1 = np.dot(K, np.eye(3, 4))
    M2 = np.dot(K, transf)
    P = linear_triangulation(P1, P2, M1, M2)
    
    left_img = cv2.imread("../Exercise 6 - Two-view Geometry/data/0001.jpg")
    right_img = cv2.imread("../Exercise 6 - Two-view Geometry/data/0002.jpg")
    rep_p1 = np.dot(M1, P)
    rep_p1 /= rep_p1[2, :]
    rep_p1 = rep_p1[:2, :].astype(np.int)
    for i in range(rep_p1.shape[1]):
        cv2.drawMarker(left_img, tuple(rep_p1[:, i]), 0xFF0000, thickness=5)
    cv2.imshow("left", left_img)
    rep_p2 = np.dot(M2, P)
    rep_p2 /= rep_p2[2, :]
    rep_p2 = rep_p2[:2, :].astype(np.int)
    for i in range(rep_p2.shape[1]):
        cv2.drawMarker(right_img, tuple(rep_p2[:, i]), 0xFF0000, thickness=5)
    cv2.imshow("right", right_img)
    cv2.waitKey(0)

    # fig = plt.figure()
    # ax1 = Axes3D(fig)

    # ax1.scatter3D(P[0, :], P[1, :], P[2,:], cmap='Blues')
    # ax1.scatter3D(0, 0, 0, cmap="reds")
    # plt.show()

    # [[9.94666151e-01 - 9.92804895e-03  1.02667822e-01  9.94713715e-01]
    #  [-1.02525787e-01  1.38908945e-02  9.94633353e-01 - 1.02686983e-01]
    #  [-1.13009165e-02 - 9.99854228e-01  1.27989216e-02 - 9.34531709e-05]]
