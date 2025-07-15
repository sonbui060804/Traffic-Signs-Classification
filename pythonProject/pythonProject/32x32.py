import cv2
import os

src_folder = 'dataset'
dst_folder = 'du_lieu_32x32'

for class_name in os.listdir(src_folder):
    src_path = os.path.join(src_folder, class_name)
    dst_path = os.path.join(dst_folder, class_name)
    os.makedirs(dst_path, exist_ok=True)

    for img_name in os.listdir(src_path):
        img = cv2.imread(os.path.join(src_path, img_name))
        img_resized = cv2.resize(img, (32, 32))
        cv2.imwrite(os.path.join(dst_path, img_name), img_resized)
