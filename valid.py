# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'


if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import os
os.environ['WANDB_DISABLED'] = 'true'

import torch
from ultralytics import YOLO
import argparse


def valid_seg_yolo(args):
    # Load a model
    model = YOLO(args.weights)

    metrics = model.val(
        data=args.dataset_config,
        project='yolo_seg_x_proj_{}'.format(args.image_size),
        imgsz=args.image_size,
        batch=1,
        iou=args.iou,
        conf=args.conf,
        half=True,
        save_json=True,
        save_txt=True,
        save_conf=True,
        # save_hybrid=True,
        plots=True,
    )
    print(metrics)
    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # a list contains map50-95 of each category


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
        "--weights",
        type=str,
        default=code_path + 'weights/yolov8x_AlphaDent_9_classes_640px.pt',
        help="Path to file with weights (.pt)"
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
        "--iou",
        type=int,
        default=0.5,
        help="Intersection over Union for NMS"
    )
    parser.add_argument(
        "--conf",
        type=int,
        default=0.001,
        help="Save all boxes with confidence larger than this value"
    )

    args = parser.parse_args()
    print('Input arguments:', args)

    valid_seg_yolo(args)
