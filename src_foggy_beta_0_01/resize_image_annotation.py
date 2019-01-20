import cv2
import os

orig_dataset_path = '../../datasets/Cityscapes/JPEGImages/'
resize_dataset_path = '../../datasets/Cityscapes/Resize_JPEGImages/'

image_names = os.listdir(orig_dataset_path)
#print(image_names)

resized_img_height = 300
resized_img_width = 600

for image in image_names:
    img = cv2.imread(os.path.join(orig_dataset_path, image))
    resized_img = cv2.resize(img, dsize=(resized_img_width, resized_img_height))
    cv2.imwrite(os.path.join(resize_dataset_path, image), resized_img)
