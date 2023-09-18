import os
import sys
import cv2
import numpy as np
import argparse
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, './'))
sys.path.append(os.path.join(CURRENT_DIR, './utils'))
from utils.file_utils import MkdirSimple

from utils.file_utils import get_files

def get_parameter():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', default=None, type=str,
                        help='Directory to save inference results and test results')

    parser.add_argument("--error_map_dir", default=None, type=str, help='Dir to save image error map')
    parser.add_argument("--save_dir", default=None, type=str, help='Dir to save image error map')

    return parser.parse_args()

def WriteResultPng(error_map, output_dir, op):
    error_img = cv2.imread(error_map,-1)
    mask = error_img.copy()
    width = mask.shape[0]
    print(mask.shape)
    mask[:width,:] = 0
    mask[width//2:,:] = 1
    error_img = error_img * mask
    err_points = np.sum(error_img>0)
    err_points_big_5 = np.sum(error_img>5)
    err_points_big_10 = np.sum(error_img>10)
    write = "number:{},     val>5: {},     val>10: {}".format(err_points,err_points_big_5 , err_points_big_10)
    cv2.putText(error_img, write, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,255,255),1,  cv2.LINE_AA)
    err_points_big_5 = err_points_big_5 / err_points
    err_points_big_10 = err_points_big_10 / err_points
    write = "ratio:    val>5: {:.2%},     val>10: {:.2%}".format(err_points_big_5, err_points_big_10)
    cv2.putText(error_img, write, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    name = os.path.splitext(op)[0]
    output_depth = os.path.join(output_dir, "comapre", name + ".png")
    MkdirSimple(output_depth)
    cv2.imwrite(output_depth, error_img)

def get_valid_image(error_map):
    error_img = cv2.imread(error_map,-1)
    mask = error_img.copy()
    width = mask.shape[0]

    mask[:width,:] = 0
    mask[width//2:,:] = 1
    error_img = error_img * mask
    err_points = np.sum(error_img>0)

    err_points_big_10 = np.sum(error_img>10)

    err_points_big_10 = err_points_big_10 / err_points

    if err_points_big_10 > 0.25:
        return False
    else:
        return True


if __name__ == '__main__':
    args = get_parameter()
    input_dir = args.error_map_dir

    output_dir = args.output_dir

    error_map_lists = get_files(input_dir)

    root_len = len(input_dir)

    for error_map in zip(sorted(error_map_lists)):
        error_map = error_map[0]
        if (error_map[root_len] == '/'):
            op = error_map[root_len + 1:]
        else:
            op = error_map[root_len:]
        if(args.save_dir is None and args.output_dir is not None):
            WriteResultPng(error_map, output_dir, op)
        elif args.save_dir is not None and args.output_dir is None:
            valid_image = get_valid_image(error_map)
            MkdirSimple(args.save_dir)

            if valid_image:
                with open(args.save_dir + "/valid_list.txt",'a') as f:
                    f.write(error_map)
                    f.write("\n")
            else:
                with open(args.save_dir + "/invalid_list.txt", 'a') as f:
                    f.write(error_map)
                    f.write("\n")