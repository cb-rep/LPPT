from __future__ import print_function
import os

import pandas as pd
from PIL import Image
import contextlib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from mod_imagefolder import CustomImageFolder
from utils2.config_utils import load_yaml, get_args
from utils2.lr_schedule import cosine_decay, adjust_lr
from models.builder import MODEL_GETTER
from eval import suppression
from torchvision import transforms
import warnings
from test_model import test_model_final

warnings.simplefilter(action='ignore', category=Warning)


# NOTE 第一次训练，获得一致认可度文件，用于分割数据集
def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((510, 510), Image.BILINEAR),
        transforms.RandomCrop((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # NOTE 自定义dataset方法
    trainset = CustomImageFolder(root='./dataset/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    # NOTE 自定义模型
    args = get_args()
    load_yaml(args, args.c)

    net = MODEL_GETTER[args.model_name](
        use_fpn = args.use_fpn,
        fpn_size = args.fpn_size,
        use_selection = args.use_selection,
        num_classes = args.num_classes,
        num_selects = args.num_selects,
        use_combiner = args.use_combiner,
    )

    # NOTE 自定义优化器
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.max_lr, nesterov=True, momentum=0.9, weight_decay=args.wdecay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.max_lr)

    schedule = cosine_decay(args, len(trainloader))
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    netp = torch.nn.DataParallel(net, device_ids=[0])
    # GPU
    device = torch.device("cuda:0")
    net.to(device)

    max_val_acc = 0

    # NOTE 记录每个epoch的类别认可数
    count = pd.DataFrame(columns=range(200))
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        optimizer.zero_grad()
        temperature = 0.5 ** (epoch // 10) * args.temperature
        n_left_batchs = len(trainloader) % args.update_freq

        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets, pos, pos2) in enumerate(trainloader):
            iterations = epoch * len(trainloader) + batch_idx
            adjust_lr(iterations, optimizer, schedule)

            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            with amp_context():
                outs = netp(inputs)
                output = outs['comb_outs']

                loss = 0.
                for name in outs:

                    if "FPN1_" in name:
                        if args.lambda_b0 != 0:
                            aux_name = name.replace("FPN1_", "")
                            gt_score_map = outs[aux_name].detach()
                            thres = torch.Tensor(net.selector.thresholds[aux_name])
                            gt_score_map = suppression(gt_score_map, thres, temperature)
                            logit = F.log_softmax(outs[name] / temperature, dim=-1)
                            loss_b0 = nn.KLDivLoss()(logit, gt_score_map)
                            loss += args.lambda_b0 * loss_b0
                        else:
                            loss_b0 = 0.0

                    elif "select_" in name:
                        if not args.use_selection:
                            raise ValueError("Selector not use here.")
                        if args.lambda_s != 0:
                            S = outs[name].size(1)
                            logit = outs[name].view(-1, args.num_classes).contiguous()
                            loss_s = nn.CrossEntropyLoss()(logit,
                                                           targets.unsqueeze(1).repeat(1, S).flatten(0))
                            loss += args.lambda_s * loss_s
                        else:
                            loss_s = 0.0

                    elif "drop_" in name:
                        if not args.use_selection:
                            raise ValueError("Selector not use here.")

                        if args.lambda_n != 0:
                            S = outs[name].size(1)
                            logit = outs[name].view(-1, args.num_classes).contiguous()
                            n_preds = nn.Tanh()(logit)
                            labels_0 = torch.zeros([batch_size * S, args.num_classes]) - 1
                            labels_0 = labels_0.to("cuda:0")
                            loss_n = nn.MSELoss()(n_preds, labels_0)
                            loss += args.lambda_n * loss_n
                        else:
                            loss_n = 0.0

                    elif "layer" in name:
                        if not args.use_fpn:
                            raise ValueError("FPN not use here.")
                        if args.lambda_b != 0:
                            loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), targets)
                            loss += args.lambda_b * loss_b
                        else:
                            loss_b = 0.0

                    elif "comb_outs" in name:
                        if not args.use_combiner:
                            raise ValueError("Combiner not use here.")

                        if args.lambda_c != 0:
                            loss_c = nn.CrossEntropyLoss()(outs[name], targets)
                            loss += args.lambda_c * loss_c

                    elif "ori_out" in name:
                        loss_ori = F.cross_entropy(outs[name], targets)
                        loss += loss_ori

                if batch_idx < len(trainloader) - n_left_batchs:
                    loss /= args.update_freq
                else:
                    loss /= n_left_batchs

            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.update_freq == 0 or (batch_idx + 1) == len(trainloader):
                if args.use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print(
                    'Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        batch_idx, loss, 100. * float(correct) / total, correct, total))

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

        train_acc = 100. * float(correct) / total
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f |\n' % (
                    epoch, train_acc, loss))

        val_acc_com = test_model_final(net)
        if val_acc_com > max_val_acc:
            max_val_acc = val_acc_com
            net.cpu()
            # torch.save(net, './' + store_name + '/model_acc{}.pth'.format(max_val_acc))
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)
        
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc = %.5f\n' % (
                epoch, val_acc_com))

        # NOTE 输出一下count
        count.to_excel('./' + store_name + '/每张图片认可�?.xlsx')


if __name__ == '__main__':
    train(nb_epoch=100,  # number of epoch
          batch_size=16,  # batch size
          store_name='first_train',  # folder for output
          resume=False,  # resume training from checkpoint
          start_epoch=0,  # the start epoch number when you resume the training
          model_path='')  # the saved model where you want to resume the training
