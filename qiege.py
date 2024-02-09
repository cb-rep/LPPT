split = {}


def qiege1():
    f = open('train2.txt', 'w')
    with open('images.txt') as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            image_id, image_path = line.split(' ')
            label = image_path.split('.')[0]
            # label=(int)label
            # label = (int)label + 1
            tmp = image_path.split('.')[1]
            tmp = tmp + '.jpg'
            image_name = image_path.split('/')[1]
            print(image_path)
            line = fp.readline()
            f.write(image_path + ' ' + label + "\n")
    f.close()


def qiege2():
    with open('train_test_split.txt') as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            image_id, image_split = line.split(' ')
            split[int(image_id)] = 'train' if int(image_split) else 'test'
            line = fp.readline()
    print(split)


if __name__ == "__main__":
    qiege2()
    f_train = open('train2.txt', 'w')
    f_test = open('test2.txt', 'w')
    with open('images.txt') as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            image_id, image_path = line.split(' ')
            label = image_path.split('.')[0]
            tmp = int(label) - 1
            tmp2 = str(tmp)
            tmp3 = tmp2.zfill(3)
            image_split = split[int(image_id)]
            if image_split == 'train':
                f_train.write(image_path + ' ' + tmp3 + "\n")
            else:
                f_test.write(image_path + ' ' + tmp3 + "\n")
            line = fp.readline()
    f_train.close()
    f_test.close()

