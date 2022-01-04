import cv2
import os


factor = 4

src_dir = '../datasets/training_hr_images/training_hr_images/'
tar_dir_hr = '../dataset/train/HRx4/'
tar_dir_lr = '../dataset/train/LRx4/'

for file in os.listdir(src_dir):
    img = cv2.imread(src_dir + file)
    height, width, channels = img.shape

    img = cv2.hconcat([img] * factor)
    img = cv2.vconcat([img] * factor)

    LR_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(tar_dir_hr + file, img)
    cv2.imwrite(tar_dir_lr + file, LR_img)
