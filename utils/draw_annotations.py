# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), AlphaChip'


import glob
import os
import cv2
import numpy as np
import argparse


def load_yolo_annotations(annotation_file, image_width, image_height):
    """
    Loads YOLO annotations for instance segmentation from a file.

    :param annotation_file: Path to the annotation file.
    :param image_width: Width of the image.
    :param image_height: Height of the image.
    :return: A list of annotations for each image.
    """
    annotations = []
    with open(annotation_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])

            # Read the coordinates of the object mask (polygons)
            mask_coords = []
            x_min = 100000000
            y_min = 100000000
            x_max = -100000000
            y_max = -100000000
            for i in range(1, len(parts), 2):
                mask_coords.append((float(parts[i]) * image_width, float(parts[i + 1]) * image_height))
                if mask_coords[-1][0] < x_min:
                    x_min = mask_coords[-1][0]
                if mask_coords[-1][1] < y_min:
                    y_min = mask_coords[-1][1]
                if mask_coords[-1][0] > x_max:
                    x_max = mask_coords[-1][0]
                if mask_coords[-1][1] > y_max:
                    y_max = mask_coords[-1][1]

            annotations.append((class_id, int(x_min), int(y_min), int(x_max), int(y_max), mask_coords))

    return annotations


def visualize_yolo_instance_segmentation(store_folder, image_path, annotation_file):
    """
    Visualizes YOLO annotations for instance segmentation on an image.

    :param image_path: Path to the image.
    :param annotation_file: Path to the YOLO annotation file.
    """
    # Load the image
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Load annotations
    annotations = load_yolo_annotations(annotation_file, image_width, image_height)

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Lime Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 255, 128),  # Spring Green / Teal
        (128, 255, 0)  # Chartreuse
    ]

    for ann in annotations:
        class_id, x_min, y_min, x_max, y_max, mask_coords = ann

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[class_id % 10], 10)

        # Draw the mask (polygons) for each object
        mask_points = np.array(mask_coords, dtype=np.int32)
        cv2.polylines(image, [mask_points], isClosed=True, color=colors[class_id % 10], thickness=10)

        # Add text with the class
        cv2.putText(image, f"Class {class_id}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 5.0, colors[class_id % 10], 10)

    # Display the image
    cv2.imwrite(store_folder + os.path.basename(image_path), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to original AlphaDent dataset",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to directory to store images with visualized annotations",
        required = True,
    )
    args = parser.parse_args()
    print('Input arguments:', args)
    os.makedirs(args.output_path, exist_ok=True)

    images = glob.glob(args.input_path + '/images/train/*.jpg')
    labels = []
    for img_path in images:
        lbl_path = args.input_path + '/labels/train/' + os.path.basename(img_path)[:-4] + '.txt'
        labels.append(lbl_path)

    for i in range(len(images)):
        image_path = images[i] # Path to the image
        annotation_file = labels[i]  # Path to the YOLO annotation file (instance segmentation)
        print(annotation_file)
        print(image_path)
        visualize_yolo_instance_segmentation(args.output_path, image_path, annotation_file)