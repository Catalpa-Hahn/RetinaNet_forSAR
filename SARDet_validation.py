import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

# assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', default='../Datasets/SARDet-100K', help='Path to COCO directory')
    parser.add_argument('--model_path', default='./weights/R50_154.pt', help='Path to model', type=str)
    parser.add_argument('--save_path', default='./runs/val/R50', help='结果存储路径', type=str)
    parser.add_argument('--depth', default=50, type=int, help='Resnet depth, must be one of 18, 34, 50, 101, 152')
    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='test',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_val.num_classes(), pretrained=True)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        # retinanet.load_state_dict(torch.load(parser.model_path).module.state_dict())
        retinanet.load_state_dict(torch.load(parser.model_path).state_dict(), strict=False)
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet, json_root=parser.save_path)


if __name__ == '__main__':
    main()
