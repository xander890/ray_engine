import numpy as np

def norm2_diff(img1, img2):
    diff = img2 - img1
    norm2 = np.square(diff[:,:,0])+ np.square(diff[:,:,1]) + np.square(diff[:,:,2])
    return norm2

def rmse(img1, img2):
    norm2 = norm2_diff(img1, img2)
    norm2_sum = np.sum(norm2[:])
    return np.sqrt(norm2_sum) / norm2.size