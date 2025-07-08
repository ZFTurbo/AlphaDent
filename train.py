# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'


import os
os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import torch
from ultralytics import YOLO
import argparse


def train_seg_yolo(args):
    # Load a model
    model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=args.dataset_config,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        project='yolo_seg_x_proj_{}'.format(args.image_size),
        deterministic=False,
        plots=True,
        device=[0],
    )

    print(results)


if __name__ == '__main__':
    print('Torch: {} Cuda is available: {}'.format(torch.__version__, torch.cuda.is_available()))
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=code_path + 'AlphaDent/yolo_seg_train.yaml',
        help="Path to yolo_seg_train.yaml for AlphaDent dataset"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=640,
        help="Image size in pixels for model input"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for model. Set according to your GPU memory"
    )

    args = parser.parse_args()
    print('Input arguments:', args)

    train_seg_yolo(args)
