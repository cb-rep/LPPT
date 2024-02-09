import torch

from models.builder import MODEL_GETTER
from data.dataset import build_loader
from utils2.config_utils import load_yaml, build_record_folder, get_args
from eval import evaluate, cal_train_metrics


def test_model(model_path):
    args = get_args()
    load_yaml(args, './configs/config.yaml')
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     transform_test = transforms.Compose([
    #         transforms.Resize((510, 510), Image.BILINEAR),
    #         transforms.CenterCrop((384, 384)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225]
    #         ),
    #     ])

    #     testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                                transform=transform_test)
    #     val_loader = torch.utils.data.DataLoader(testset, num_workers=1, shuffle=True, batch_size=16)
    train_loader, val_loader = build_loader(args)

    # 原本
    # model = torch.load(model_path)

    model = MODEL_GETTER[args.model_name](
        use_fpn=args.use_fpn,
        fpn_size=args.fpn_size,
        use_selection=args.use_selection,
        num_classes=args.num_classes,
        num_selects=args.num_selects,
        use_combiner=args.use_combiner,
    )  # about return_nodes, we use osur default setting
    checkpoint = torch.load('./best.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    device = torch.device("cuda:0")
    model.to(device)

    # GPU
    # device = torch.device("cuda:0")
    # model.to(device)


def test_model_final(model):
    args = get_args()
    load_yaml(args, './configs/config.yaml')
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     transform_test = transforms.Compose([
    #         transforms.Resize((510, 510), Image.BILINEAR),
    #         transforms.CenterCrop((384, 384)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225]
    #         ),
    #     ])

    #     testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                                transform=transform_test)
    #     val_loader = torch.utils.data.DataLoader(testset, num_workers=1, shuffle=True, batch_size=16)
    train_loader, val_loader = build_loader(args)

    # GPU
    device = torch.device("cuda:0")
    model.to(device)

    acc, eval_name, eval_acces = evaluate(args, model, val_loader)
    # print(acc, eval_name, eval_acces)
    return acc


if __name__ == '__main__':
    test_model('./second_train/model.pth')
