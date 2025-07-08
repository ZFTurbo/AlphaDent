# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), AlphaChip'

import glob
import argparse
import shutil


if __name__ == '__main__':
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
        help="Path to directory to store updated dataset for 4 classes",
        required = True,
    )
    args = parser.parse_args()
    print('Input arguments:', args)

    inp_folder = args.input_path + '/'
    out_folder = args.output_path + '/'

    shutil.copytree(inp_folder, out_folder)

    # Replace txt files
    txt_paths = glob.glob(out_folder + '**/*.txt', recursive=True)
    for txt_path in txt_paths:
        lines = open(txt_path).readlines()
        out = open(txt_path, 'w')
        for line in lines:
            if line[0] == '4' or line[0] == '5' or line[0] == '6' or line[0] == '7' or line[0] == '8':
                out.write('3' + line[1:])
            else:
                out.write(line)
        out.close()

    id_to_classes = {
        1: 'Abrasion',
        2: 'Filling',
        3: 'Crown',
        4: 'Caries',
    }

    # Create .yaml file
    out = open(out_folder + 'yolo_seg_train.yaml', 'w')
    out.write('path: {}\n'.format(out_folder))
    out.write('train: images/train\n')
    out.write('val: images/valid\n')
    out.write('names:\n')
    out.write('  0: Abrasion\n')
    out.write('  1: Filling\n')
    out.write('  2: Crown\n')
    out.write('  3: Caries\n')
    out.close()
