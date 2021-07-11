import cv2
import numpy as np

# ex1

def generate_corner_points(square_size=0.04, board_size=(6, 9)):
    points = []
    for w in range(board_size[0]):
        for h in range(board_size[1]):
            points.append([h * square_size, w * square_size, 0])
    return points

def get_rotation_matrix_from_angles(angles):
    w = np.asarray(angles).T
    theta = np.linalg.norm(w)
    k = w / theta
    k_cross = np.mat([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * k_cross + (1 - np.cos(theta)) * (k_cross * k_cross)
    return R

def draw_cube(img, corners):

    edges = ((0, 1), (0, 2), (1, 3), (2, 3),
             (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7))
    for (p1, p2) in edges:
        cv2.line(img, corners[p1], corners[p2], color=(0, 0, 255), thickness=2)

def read_params_from_txt(K_file, D_file):
    K, D = [], []
    with open(K_file) as f:
        for line in f.readlines():
            K.append([float(i) for i in line.strip().split(' ') if i])
    with open(D_file) as f:
        for line in f.readlines():
            D = [float(i) for i in line.strip().split(' ') if i]
    return K, D

# ex2

def read_K_from_txt(K_file):
    K = []
    with open(K_file) as f:
        for line in f.readlines():
            K.append([float(i) for i in line.strip().split(' ') if i])
    return K

def read_P_w_from_txt(pw_file):
    P_w = []
    with open(pw_file) as f:
        for line in f.readlines():
            P_w.append([float(i) for i in line.strip().split(',') if i])
    return np.asarray(P_w) / 100.

def read_p_wave_from_txt(pwave_file):
    p_wave = []
    with open(pwave_file) as f:
        for line in f.readlines():
            p_wave.append([float(i) for i in line.strip().split(' ') if i])
    p_wave = np.asarray(p_wave)
    return np.asarray(p_wave).reshape((-1, 12, 2))