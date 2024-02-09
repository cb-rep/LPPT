# NOTE 根据一致认可度分割数据集
# NOTE 第一次分割数据集
import os
import shutil
import numpy as np
import pandas as pd

img_path = 'dataset/train'
a1_path = './first/a1'
b1_path = './first/b1'

if not os.path.exists(a1_path):
    os.makedirs(a1_path)
if not os.path.exists(a1_path):
    os.makedirs(b1_path)

# 记录每个类别的文件夹已经放入几张图片，方便后续图片名字重新修改
img_num1 = np.zeros(200)  # a1数据集
img_num2 = np.zeros(200)  # b1数据集
img_num1 = img_num1.astype(int)
img_num2 = img_num2.astype(int)


# 把图片粘贴至数据集
def copy_one_img(class_number, image_number, label, is_a):  # 类别名称，在类别里的序号，一致认可度高的标签，放入a还是b数据集
    # 图片路径
    source_file_path = img_path + '/class_' + class_number + '/' + image_number + '.jpg'
    formatted_number = '{:03d}'.format(label)

    # 如果放入a数据集
    if is_a:
        destination_folder_path = './first/a1' + '/class_' + formatted_number
        # 判断文件夹是否存在
        if not os.path.exists(destination_folder_path):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(destination_folder_path)

        new_file_name = str(img_num1[label - 1] + 1) + '.jpg'
        # 目标文件的完整路径
        destination_file_path = os.path.join(destination_folder_path, new_file_name)
        # 复制文件并修改文件名
        shutil.copyfile(source_file_path, destination_file_path)
        # 这个类别文件夹图片数+1
        img_num1[label - 1] += 1
    else:  # 放入b数据集
        destination_folder_path = './first/b1' + '/class_' + formatted_number
        if not os.path.exists(destination_folder_path):
            os.makedirs(destination_folder_path)

        new_file_name = str(img_num2[label - 1] + 1) + '.jpg'
        destination_file_path = os.path.join(destination_folder_path, new_file_name)
        shutil.copyfile(source_file_path, destination_file_path)
        img_num2[label - 1] += 1


def get_max_index(row):
    max_index = row.iloc[:].idxmax()
    return max_index


if __name__ == '__main__':
    df = pd.read_excel('./first_train/每张图片认可数.xlsx')
    # 新增label列
    df2 = df.iloc[:, 1:201]
    df2['label'] = df2.apply(get_max_index, axis=1)
    df['label'] = df2['label']  # 一致认可度高的标签
    df.to_excel('./first_train/每张图片认可数2.xlsx')
    # 记录分到a1数据集的每一类别有几张
    num3 = np.zeros(200)
    num3 = num3.astype(int)

    for i in range(len(df)):
        pos = df.iloc[i, 0]
        split_string = pos.split('-')
        class_number = split_string[0]  # 类别名称
        image_number = split_string[1]
        # 把错误次数小于2的分到a1数据集 并且一致认可度高的标签等于真实标签
        if df.iloc[i, 203] < 2 and int(class_number) == df.iloc[i, 204] + 1:
            copy_one_img(class_number, image_number, df.iloc[i, 204] + 1, 1)  # df.iloc[i, 204] + 1是一致性认可度高的标签
        else:
            copy_one_img(class_number, image_number, df.iloc[i, 204] + 1, 0)
