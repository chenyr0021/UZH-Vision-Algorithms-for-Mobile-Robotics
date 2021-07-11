import utils
import cv2
import numpy as np

class PerspectiveProjector:
    def __init__(self, K, D):
        self.K = np.array(K)
        self.D = np.array(D)

    def coordinate_transform(self, R, t, pw):
        # print(pw)
        pw_4d = np.array([pw + [1]])
        # print(pw_4d)
        R = np.mat(R)
        t = np.mat(t).T
        transform_mat = np.concatenate((R, t), axis=1)
        return (transform_mat * pw_4d.T).T.tolist()

    def perspective_projection(self, R, t, pw):
        pc = self.coordinate_transform(R, t, pw)
        pixel_point = np.dot(self.K, np.array(pc).T).T.squeeze()
        pixel_point = pixel_point/pixel_point[2]
        pixel_point = np.rint(pixel_point).astype('int').tolist()
        return pixel_point[:2]

    def distort_point(self, p):
        principle = np.array([self.K[0, 2], self.K[1, 2]])
        p = np.array(p)
        r_2 = np.dot((p - principle), (p - principle))
        p_d = (1 + self.D[0] * r_2 + self.D[1] * r_2**2) * (p - principle) + principle
        return p_d

    def distort_point_vectorized(self, P):
        principle = np.array([self.K[0, 2], self.K[1, 2]])
        P = np.asarray(P)
        R_2 = np.sum(np.multiply(P - principle, P - principle), axis=1).reshape((-1, 1))
        P_d = np.multiply((1 + self.D[0] * R_2 + self.D[1] * R_2**2), (P - principle)) + principle
        return P_d

    def undistort_image(self, img):
        [h, w, c] = img.shape
        undistort_img = np.zeros(img.shape, dtype=np.uint8)
        for j in range(w):
            for i in range(h):
                u, v =self.distort_point([j, i])
                u1, v1 = np.floor([u, v]).astype('int').tolist()
                # a, b = u - u1, v - v1
                # if u1 + 1 >= 0 and u1 + 1 < w and v1 + 1 >= 0 and v1 + 1 < h:
                #     undistort_img[i, j] = (1 - b) * ((1 - a) * img[v1, u1] + a * img[v1, u1 + 1]) \
                #                             + b * ((1 - a) * img[v1 + 1, u1] + a * img[v1 + 1, u1 + 1])
                undistort_img[i, j, :] = img[v1, u1, :]
        return undistort_img

    # TODO:
    def undistort_image_vectorized(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        [h, w] = gray_img.shape
        undistort_img = np.zeros(gray_img.shape, dtype=np.uint8).reshape((1, -1))
        P = []
        for j in range(w):
            for i in range(h):
                P.append([j, i])
        P = np.asarray(P)
        P_d = self.distort_point_vectorized(P)
        P_d = np.rint(P_d).astype(np.uint32)
        flatted_P_d = P_d[:, 0] + P_d[:, 1] * w
        flatted_P = P[:, 0] + P[:, 1] * w
        resized_img = np.resize(gray_img, (1, h*w))
        print(flatted_P_d.shape)
        undistort_img[0, flatted_P] = resized_img[0, flatted_P_d]
        return cv2.cvtColor(undistort_img.reshape((h, w)), cv2.COLOR_GRAY2BGR)




if __name__ == '__main__':
    # read K D from file
    K_file = "E:/chenyr/Desktop/VSLAM/Exercise 1 - Augmented Reality Wireframe Cube/data/K.txt"
    D_file = "E:/chenyr/Desktop/VSLAM/Exercise 1 - Augmented Reality Wireframe Cube/data/D.txt"
    pose_file = "E:/chenyr/Desktop/VSLAM/Exercise 1 - Augmented Reality Wireframe Cube/data/poses.txt"
    undistort_img_file = "E:/chenyr/Desktop/VSLAM/Exercise 1 - Augmented Reality Wireframe Cube/data/images_undistorted/img_0001.jpg"
    distort_img_file = "E:/chenyr/Desktop/VSLAM/Exercise 1 - Augmented Reality Wireframe Cube/data/images/img_0001.jpg"
    images_dir = "E:/chenyr/Desktop/VSLAM/Exercise 1 - Augmented Reality Wireframe Cube/data/images/"
    chessboard_square_size = 0.04

    # Part I

    undistort_img = cv2.imread(undistort_img_file)
    distort_img = cv2.imread(distort_img_file)

    K, D = utils.read_params_from_txt(K_file, D_file)
    angles = None
    t = None

    with open(pose_file) as f:
        line = f.readline().strip().split(" ")
        angles = [float(i) for i in line[:3]]
        t = [float(i) for i in line[3:]]

    R = utils.get_rotation_matrix_from_angles(angles)

    projector = PerspectiveProjector(K, D)
    pw_points = utils.generate_corner_points()

    # show chessboard corners 1
    for p in pw_points:
        img_p = projector.perspective_projection(R, t, p)
        # print(img_p)
        cv2.circle(undistort_img, img_p, radius=3, color=(0, 0, 255), thickness=-1)

    cube_start_point = [0, 0, 0]
    factor = 2
    cube_corners = []
    for z in range(0, -2, -1):
        for y in range(2):
            for x in range(2):
                cube_corners.append([(x + cube_start_point[0]) * chessboard_square_size * factor,
                                (y + cube_start_point[1]) * chessboard_square_size * factor,
                                (z + cube_start_point[2]) * chessboard_square_size * factor])

    pixel_cube_corners = []
    for c in cube_corners:
        pixel_cube_corners.append(projector.perspective_projection(R, t, c))
    utils.draw_cube(undistort_img, pixel_cube_corners)


    # cv2.imshow("distort_img", distort_img)
    # cv2.imshow("undistort_img", undistort_img)
    # cv2.waitKey(0)

    # ----------------- Part II ------------------- #

    for p in pw_points:
        img_p = projector.perspective_projection(R, t, p)
        img_p1 = projector.distort_point(img_p).astype('int')
        cv2.circle(distort_img, img_p1, radius=3, color=(0, 0, 255), thickness=-1)

    corrected_img = projector.undistort_image_vectorized(distort_img)
    cv2.imshow("corrected_img", corrected_img)
    cv2.waitKey(0)

    # R = []
    # t = []
    # with open(pose_file) as f:
    #     for line in f.readlines():
    #         line = line.strip().split(" ")
    #         angles = [float(i) for i in line[:3]]
    #         R.append(get_rotation_matrix_from_angles(angles))
    #         t.append([float(i) for i in line[3:]])
    #
    # img_files = os.listdir(images_dir)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # h, w, c = undistort_img.shape
    # video_writer = cv2.VideoWriter('cube.avi',fourcc=fourcc, fps=30.0, frameSize=(w, h), isColor=True)
    # for i, f in enumerate(img_files):
    #     distort_img = cv2.imread(os.path.join(images_dir, f))
    #     corrected_img = projector.undistort_image(distort_img)
    #     pixel_cube_corners = []
    #     for c in cube_corners:
    #         pixel_cube_corners.append(projector.perspective_projection(R[i], t[i], c))
    #     draw_cube(corrected_img, pixel_cube_corners)
    #     video_writer.write(corrected_img)
    #     print("write frame: ", f)
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break
    #
    # video_writer.release()

