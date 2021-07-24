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
    for s in range(-1, S + 2):
        sigma = np.power(2, s/S) * sigma0
        blurred_img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, borderType=cv2.BORDER_DEFAULT)
        blurred_imgs[s, :, :] = blurred_img
        # print("guassian.shape: ", blurred_imgs.shape)
    return blurred_imgs

def DoG(blurred_imgs):
    s, h, w = blurred_imgs.shape
    shifted_imgs = np.concatenate([blurred_imgs[1:, :, :], np.zeros((1, h, w))], axis=0)
    dog = (shifted_imgs - blurred_imgs)[:-1, :, :]
    # print("dog.shape", dog.shape)
    return dog

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
            coords = np.argwhere(ext[descriptor_rad:-descriptor_rad, descriptor_rad:-descriptor_rad] == 1)
            # coords[:, 0] = (coords[:, 0] + 1) * np.power(2, octave) - 1
            # coords[:, 1] = (coords[:, 1] + 1) * np.power(2, octave) - 1
            # print(np.sum(np.array(ext, dtype=np.int)), h * w)
            # cv2.imshow("b", np.array(ext, dtype=np.float) * 255)
            # cv2.waitKey(0)
            cur_octave_extrema.append(coords)
        extrema.append(cur_octave_extrema)
    return extrema

def generate_descriptors(Gau_pyramid, extrema_loc):
    Gaussian_kernel = cv2.getGaussianKernel(16, 1.5 * 16)
    descriptors = []
    for octave in range(len(extrema_loc)):
        for scale in range(len(extrema_loc[0])):
            dx, dy = cv2.spatialGradient((Gau_pyramid[octave][scale-1, :, :] * 255).astype('uint8'))
            # change to np.int to avoid overflow
            norm = np.sqrt(dx.astype('int')**2 + dy.astype('int')**2)
            # cv2.imshow("a", norm)
            # cv2.waitKey(0)
            # print(dx.shape)
            for (x, y) in extrema_loc[octave][scale]:
                # print(x, y)
                weighted_norm_patch = norm[x-7:x+9, y-7:y+9] * Gaussian_kernel
                # get angle and transform to (0, 2*pi)
                gradient_angle = np.arctan2(dy[x-7:x+9, y-7:y+9], dx[x-7:x+9, y-7:y+9]) + np.pi
                # print(gradient_angle)
                # exit()
                descriptors.append(gradient_histogram(gradient_angle, weighted_norm_patch))
                
    return np.asarray(descriptors)
    
def match_SIFT_descriptors(des1, des2, max_ratio):
    distance = cdist(des1, des2)
    print(distance.shape)
    min_dis = np.min(distance, axis=-1)
    min_dis_ind = np.argmin(distance, axis=-1)
    distance[min_dis_ind] = 0
    
    vice_min_dis = np.min(distance, axis=-1)
    min_dis_ind[min_dis/vice_min_dis > max_ratio] = -1
    
    unique_ind = np.zeros(min_dis_ind.shape, dtype=np.int)
    _, w = np.unique(min_dis_ind, return_index=True)
    unique_ind[w] = min_dis_ind[w]
    return unique_ind
    
    
                
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
            # exit()
            descriptor[i//4, j//4, :] = normalized_HoG
    return np.resize(descriptor, (-1,))
    
def restore_keypoints_position(extrema_loc):
    image_points = []
    for octave in range(len(extrema_loc)):
        for scale in range(len(extrema_loc[0])):
            for (x, y) in extrema_loc[octave][scale]:
                image_points.append([(x + 1) * np.power(2, octave) - 1, (y + 1) * np.power(2, octave) - 1])
    return image_points


if __name__ == '__main__':
    img_dir = "../exercise4/images/"
    img1 = cv2.imread(img_dir + "img_1.jpg", cv2.IMREAD_GRAYSCALE) / 255
    img2 = cv2.imread(img_dir + "img_2.jpg", cv2.IMREAD_GRAYSCALE) / 255
    h, w = img2.shape
    img1, img2 = cv2.resize(img1, (w//2, h//2)), cv2.resize(img2, (w//2, h//2))
    
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
    # TODO: find position of descriptors in origin images

    Gau_pyramid, dog_pyramid = DoG_pyramid(img2, Octave, Scale, sigma0)
    extrema_locations2 = get_local_extremas(dog_pyramid, DoG_lower_thresh, descriptor_rad)
    descriptors2 = generate_descriptors(Gau_pyramid, extrema_locations2)
    keypoints2 = restore_keypoints_position(extrema_locations2)
    
    matches = match_SIFT_descriptors(descriptors1, descriptors2, max_ratio)
    
    
    
    
    
    
    