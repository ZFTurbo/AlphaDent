# AlphaDent
Repository for AlphaDent dataset. It contains links for dataset, train, validation and inference scripts for Yolov8.

## Dataset links

* Dataset on zenodo, kaggle, huggingface, github

## Train

* Download dataset and put in the folder with this code. Then fix path in `yolo_seg_train.yaml` if needed.
* Then you can train with following script:

```bash
python3 train.py --dataset_config ./AlphaDent/yolo_seg_train.yaml --batch_size 16 --epochs 100 --image_size 640
```

Results of training will be stored in folder `./yolo_seg_x_proj_640`.

## Pretrained weights

There are 3 different pretrained wights available: 
1) Yolo_v8x, 9 classes and 640 input. Download: [Link]()
2) Yolo_v8x, 9 classes and 960 input. Download: [Link]()
3) Yolo_v8x, 4 classes and 960 input. Download: [Link]()

## Validation

Validation will run model with validation data and output metrics.

```bash
python3 valid.py --weights './weights/yolov8x_AlphaDent_9_classes_640px.pt' --dataset_config './AlphaDent/yolo_seg_train.yaml' --batch_size 16 --epochs 100 --image_size 640
```

## Inference

If you have new dental photos for which you want to obtain predictions you can use inference script.

```bash
python3 inference.py --weights './weights/yolov8x_AlphaDent_9_classes_640px.pt' --input_path './AlphaDent/images/test/' --output_path './output/' --batch_size 16 --image_size 640
```

## Useful scripts

### Convert 9 classes dataset to 4 classes

```bash
python3 utils/convert_9_classes_dataset_to_4_classes.py --input_path './AlphaDent/' --output_path './AlphaDent_4_classes/' 
```

### Draw Yolo annotations

```bash
python3 utils/draw_annotations.py --input_path './AlphaDent/' --output_path './Draw_Annotations/' 
```

## Citations

TBD

