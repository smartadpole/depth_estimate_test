import os
import sys
import cv2
import numpy as np
import argparse
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, './'))
sys.path.append(os.path.join(CURRENT_DIR, './utils'))

from utils.file_utils import get_left_right_files
from utils.compare_predict_gt_disp import getAbsdiff,get_abs_diff_uint8
from utils.compare_tof import get_boundary_wh
from utils.file_utils import get_files, get_last_name

def get_parameter():
    parser = argparse.ArgumentParser()

    parser.add_argument('--height', default=None, type=int, help='Image height for inference')
    parser.add_argument('--width', default=None, type=int, help='Image width for inference')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='Directory to save inference results and test results')

    parser.add_argument("--disp_dir", default=None, type=str, help='Dir to save image disp')

    parser.add_argument("--tof_dir", default=None, type=str, help='Dir to save image tof 3 channels')

    parser.add_argument('--bf', type=int, default=3424, help='bf for test to generate depth')

    return parser.parse_args()

def disp_tf_errorMap(tof_file, disp_file, output_dir, op, bf=3424, width=None, height=None):
    # tof_file = "/home/indemind/delete/tof/22_1614045202151753.png"
    disp_output = cv2.imread(disp_file, -1).astype(np.float32)
    disp_output = disp_output / 256

    # compare_depth_disp("test_d", "22_1614045202151753.png", disp_output,
    #                    "/home/indemind/delete/tof/22_1614045202151753.png", width=576, height=360, bf=3424, scale=100)

    gt = cv2.imread(tof_file, -1)

    gt = gt[:, :, 0] + (gt[:, :, 1] > 0) * 255 + gt[:, :, 1] + (
            gt[:, :, 2] > 0) * 511 + gt[:, :, 2]

    if width is not None and height is not None:
        left, right, top, bottom = get_boundary_wh(gt, width=width, height=height)

        gt = gt[top: bottom, left: right]
        left, right, top, bottom = get_boundary_wh(disp_output, width=width, height=height)
        disp_output = disp_output[top: bottom, left: right]

    gt_float = gt.astype(np.float32)
    disp_output[disp_output>0] = bf / disp_output[disp_output >0]

    get_abs_diff_uint8(disp_output, gt_float, output_dir, op)

if __name__ == '__main__':
    args = get_parameter()
    input_dir = args.disp_dir
    tof_dir = args.tof_dir#"/home/indemind/delete/tof/22_1614045202151753.png"
    output_dir = args.output_dir

    root_len = len(input_dir)

    disp_files = get_files(input_dir)
    tof_files = get_files(tof_dir)
    if (len(disp_files) == 0 or len(tof_files) == 0):
        assert 0,"disp or tof file's length is 0! Stop!"
    for disp_file, tof_file in zip(sorted(disp_files), sorted(tof_files)):
        assert get_last_name(disp_file) == get_last_name(tof_file) \
            , "left_file and right_file not same"

        if (disp_file[root_len] == '/'):
            op = disp_file[root_len + 1:]
        else:
            op = disp_file[root_len:]

        disp_tf_errorMap(tof_file=tof_file, disp_file=disp_file,output_dir=output_dir,op=op, width=args.width, height=args.height)