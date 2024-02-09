import torch
from torch.autograd import Variable
from torchvision import transforms
from mod_imagefolder import CustomImageFolder2
from PIL import Image

# 把b1数据集进行分割
def test2(net, batch_size, count, epoch):
    net.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0")

    transform_test = transforms.Compose([
        transforms.Resize((510, 510), Image.BILINEAR),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    testset = CustomImageFolder2(root='./first/b1', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    with torch.no_grad():
        for batch_idx, (inputs, targets, pos, pos2, is_a) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
            outs = net(inputs)
            output = outs['comb_outs']

            _, predicted = torch.max(output.data, 1)

            # NOTE 每个batch记录数量
            if epoch < 5:
                for i in range(len(pos)):
                    row_index = pos[i]
                    col_index = predicted[i].item()
                    if row_index not in count.index:
                        count.loc[row_index] = 0
                    count.at[row_index, col_index] += 1
            else:
                for i in range(len(pos)):
                    row_index = pos[i]
                    col_index = predicted[i].item()
                    count.at[row_index, col_index] += 1

    return count