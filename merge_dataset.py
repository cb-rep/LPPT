# NOTE 合并数据集
import os
import shutil
import re

a1_path = './first/a1'
b1_path = './first/a2'


# 返回文件数量
def count_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    return len(files)


# 把图片粘贴至数据集
def copy_img():
    folders = os.listdir(b1_path)
    for i in range(len(folders)):
        folder_name = folders[i]
        class_number = re.search(r'class_(\d+)', folder_name).group(1)
        # 源图片文件夹路径
        source_folder_path = b1_path + '/class_' + class_number
        # 目标文件夹路径
        destination_folder_path = a1_path + '/class_' + class_number
        # 获得对应a数据集有几张图片
        count_a = count_files_in_folder(destination_folder_path)
        images = os.listdir(source_folder_path)
        for j in range(len(images)):
            image_name = images[j]
            # 图片路径
            source_file_path = source_folder_path + '/' + image_name
            num = count_a + 1
            count_a += 1
            # 目标文件夹路径（已修改图片名）
            destination_file_path = destination_folder_path + '/' + str(num) + '.jpg'
            shutil.copyfile(source_file_path, destination_file_path)
    return 1


if __name__ == '__main__':
    copy_img()
