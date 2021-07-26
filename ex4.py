import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.spatial.distance import cdist

# class SIFT():
#     def __init__(self, image, octave, scale, sigma0):
#         self.image = image
#         self.octave = octave
#         self.scale = scale
#         self.sigma0 = sigma0

def Guassian_blur_images(img, sigma0, S):
    h, w = img.shape
    blurred_imgs = np.zeros(( S + 3, h, w))
    for i in range(S + 3):
        s = i - 1
        sigma = np.power(2, s/S) * sigma0
        blurred_img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, borderType=cv2.BORDER_DEFAULT)
        blurred_imgs[i, :, :] = blurred_img
        # print("guassian.shape: ", blurred_imgs.shape)
        # cv2.imshow("a%d"%s, blurred_img)
    # cv2.waitKey(0)
    return blurred_imgs

def DoG(blurred_imgs):
    s, h, w = blurred_imgs.shape
    # shifted_imgs = np.concatenate([blurred_imgs[1:, :, :], np.zeros((1, h, w))], axis=0)
    dog = blurred_imgs[1:, :, :] - blurred_imgs[:-1, :, :]
    # print("dog.shape", dog.shape)
    # for i in range(s-1):
    #     show = cv2.normalize(dog[i, :, :], None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    #     cv2.imshow("a%d"%i, show)
    # cv2.waitKey(0)
    return np.abs(dog)

def DoG_pyramid(src, octave, scale, sigma0):
    h, w = src.shape
    dog_pyramid = []
    Gau_pyramid = []
    for o in range(octave):
        reshaped = cv2.resize(src, (w // np.power(2, o), h // np.power(2, o)))
        blurred = Guassian_blur_images(reshaped, sigma0, scale)
        Gau_pyramid.append(blurred)
        dog = DoG(blurred)
        dog_pyramid.append(dog)
    return Gau_pyramid, dog_pyramid

def get_local_extremas(pyramid, thresh, descriptor_rad):
    extrema = []
    # print(len(pyramid), pyramid[0].shape)
    for octave, dog in enumerate(pyramid):
        dog[dog < thresh] = 0
        s, h, w  = dog.shape
        cur_octave_extrema = []
        for i in range(1, s-1):
            maximum = maximum_filter(dog[i-1:i+2, :, :], (3, 3, 3))[1, :, :]
            ext = np.abs(maximum - dog[i, :, :]) < 1e-5
            ext *= np.abs(maximum) > 1e-5
            maximum *= ext # 保留极大值
            # non maximum suppression
            maximum = maximum_filter(maximum, (descriptor_rad, descriptor_rad))
            ext = np.abs(maximum - dog[i, :, :]) < 1e-5
            ext *= np.abs(maximum) > 1e-5
            coords = np.argwhere(ext == 1)
            cur_octave_extrema.append(coords)
        #     cv2.imshow("b%d"%i, np.array(ext, dtype=np.float) * 255)
        # cv2.waitKey(0)
        extrema.append(cur_octave_extrema)
    return extrema

def generate_descriptors(Gau_pyramid, extrema_loc):
    Gaussian_kernel = cv2.getGaussianKernel(16, 1.5 * 16)
    Gaussian_kernel = Gaussian_kernel * Gaussian_kernel.T
    descriptors = []
    for octave in range(len(extrema_loc)):
        s, h, w = Gau_pyramid[octave].shape
        for scale in range(len(extrema_loc[0])):
            dx, dy = cv2.spatialGradient((Gau_pyramid[octave][scale-1, :, :] * 255).astype('uint8'))
            # change to np.int to avoid overflow
            norm = np.sqrt(dx.astype('int')**2 + dy.astype('int')**2)
            # cv2.imshow("a", norm)
            # cv2.waitKey(0)
            # print(dx.shape)
            for (x, y) in extrema_loc[octave][scale]:
                # print(x, y)
                if 7 <= x < h-9 and 7 <= y < w-9:
                    weighted_norm_patch = norm[x-7:x+9, y-7:y+9] * Gaussian_kernel
                    # get angle and transform to (0, 2*pi)
                    gradient_angle = np.arctan2(dy[x-7:x+9, y-7:y+9], dx[x-7:x+9, y-7:y+9]) + np.pi
                    # print(gradient_angle)
                    # exit()
                    descriptors.append(gradient_histogram(gradient_angle, weighted_norm_patch))
                
    return np.asarray(descriptors)
                
def gradient_histogram(gradient_angles, weighted_norm):
    # divide to 4 * 4 cells
    descriptor = np.zeros((4, 4, 8))
    bins = [0, np.pi * 1 / 4., np.pi * 2 / 4., np.pi * 3 / 4., np.pi,
            np.pi * 5 / 4., np.pi * 6 / 4., np.pi * 7 / 4., np.pi * 2]
    for i in range(0, 16, 4):
        for j in range(0, 16, 4):
            cell = gradient_angles[i:i+4, j:j+4]
            norm_cell = weighted_norm[i:i+4, j:j+4]
            HoG, _ = np.histogram(cell, bins, weights=norm_cell)
            normalized_HoG = HoG / (np.linalg.norm(HoG) + 1e-5)
            # print(HoG)
            descriptor[i//4, j//4, :] = normalized_HoG
    return np.resize(descriptor, (128,))
    
def restore_keypoints_position(extrema_loc):
    image_points = []
    for octave in range(len(extrema_loc)):
        for scale in range(len(extrema_loc[0])):
            for (x, y) in extrema_loc[octave][scale]:
                image_points.append(((x + 1) * np.power(2, octave) - 1, (y + 1) * np.power(2, octave) - 1))
    return image_points


def match_SIFT_descriptors(des1, des2, max_ratio):
    distance = cdist(des1, des2)
    min_dis = np.min(distance, axis=-1)
    print(min_dis.shape)
    min_dis_ind = np.argmin(distance, axis=-1)
    # print(min_dis)
    # print(min_dis_ind)
    for (i, j) in enumerate(min_dis_ind):
        distance[i, j] = np.inf
    
    vice_min_dis = np.min(distance, axis=-1)
    print(min_dis / vice_min_dis)
    # min_dis_ind[min_dis / vice_min_dis > max_ratio] = -1
    
    unique_ind = np.ones(min_dis_ind.shape, dtype=np.int) * (-1)
    _, w = np.unique(min_dis_ind, return_index=True)
    unique_ind[w] = min_dis_ind[w]
    return unique_ind

if __name__ == '__main__':
    img_dir = "../exercise4/images/"
    img1 = cv2.imread(img_dir + "img_1.jpg", cv2.IMREAD_GRAYSCALE) / 255
    img2 = cv2.imread(img_dir + "img_2.jpg", cv2.IMREAD_GRAYSCALE) / 255
    h, w = img2.shape
    img1, img2 = cv2.resize(img1, (w//5, h//5)), cv2.resize(img2, (w//5, h//5))
    
    Octave = 5
    Scale = 3
    sigma0 = 1.6
    DoG_lower_thresh = 0.04
    descriptor_rad = 8
    max_ratio = 0.8
    
    Gau_pyramid, dog_pyramid = DoG_pyramid(img1, Octave, Scale, sigma0)
    extrema_locations1 = get_local_extremas(dog_pyramid, DoG_lower_thresh, descriptor_rad)
    descriptors1 = generate_descriptors(Gau_pyramid, extrema_locations1)
    keypoints1 = restore_keypoints_position(extrema_locations1)

    img1_bgr = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    img1_bgr = cv2.cvtColor(img1_bgr, cv2.COLOR_GRAY2BGR)
    

    Gau_pyramid, dog_pyramid = DoG_pyramid(img2, Octave, Scale, sigma0)
    extrema_locations2 = get_local_extremas(dog_pyramid, DoG_lower_thresh, descriptor_rad)
    descriptors2 = generate_descriptors(Gau_pyramid, extrema_locations2)
    keypoints2 = restore_keypoints_position(extrema_locations2)

    img2_bgr = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    img2_bgr = cv2.cvtColor(img2_bgr, cv2.COLOR_GRAY2BGR)
    
    
    matches = match_SIFT_descriptors(descriptors1, descriptors2, max_ratio)
    valid_keypoints = [[], []]
    for i1, i2 in enumerate(matches):
        if i2 != -1:
            valid_keypoints[0].append(i1)
            valid_keypoints[1].append(i2)
            x, y = keypoints1[i1]
            cv2.drawMarker(img1_bgr, (y, x), color=(0, 0, 255))
            x, y = keypoints2[i2]
            cv2.drawMarker(img2_bgr, (y, x), color=(0, 0, 255), markerSize=5, thickness=2)
    cv2.imshow("img1", img1_bgr)
    cv2.imshow("img2", img2_bgr)
    cv2.waitKey(0)
    # out_img = np.array((h, w))
    # cv2.drawMatches(img1_bgr, valid_keypoints[0], img2, valid_keypoints[1], matches, out_img)
    # cv2.imshow("res", out_img)
    
    
    
    
    
    
    