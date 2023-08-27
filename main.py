import argparse
import sys, os
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

from onnx_utils.onnxmodel import ONNXModel
from utils.file_utils import get_left_right_files

from utils.preprocess_postprocess import preprocess_hit, preprocess_madnet
from utils.file_utils import WriteDepth
from utils.file_utils import get_files, get_last_name
from utils.compare_tof import compare_depth_tof
from utils.compare_predict_gt_disp import compare_depth_disp

def get_parameter():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default=None, required=True, type=str, help='Data directory for prediction')

    parser.add_argument('--model_type', default="madnet", type=str, help='model type of onnx')

    parser.add_argument('--img_height', default=400, type=int, help='Image height for inference')
    parser.add_argument('--img_width', default=640, type=int, help='Image width for inference')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='Directory to save inference results and test results')

    parser.add_argument('--onnx_file', default=None, type=str,
                        help='File to save onnx inference model')

    parser.add_argument("--disp_dir", default=None, type=str, help='Dir to save image disp')

    parser.add_argument("--depth_dir", default=None, type=str, help='Dir to save image depth to compare tof')

    parser.add_argument("--tof_dir", default=None, type=str, help="Dir to save images of tof point")

    parser.add_argument("--tof_selected", default=None, type=str, help="Dir to save images of tof points selected")

    parser.add_argument('--bf', type=str, default=3424, help='bf for test to generate depth, only parker'
                                                             ' need this parameter now')

    return parser.parse_args()
def main():
    args = get_parameter()

    depth_from_onnx = False

    print("current dataset's bf is {}".format(args.bf))

    if os.path.isdir(args.data_dir):
        left_files, right_files = get_left_right_files(args.data_dir)

        # load onnx file
        model = ONNXModel(args.onnx_file)
        depth_from_onnx = True

        root_len = len(args.data_dir)

        for left_file, right_file in zip(left_files, right_files):
            if left_file[root_len:][0] == '/':
                op = os.path.join(args.output_dir, left_file[root_len + 1:])
            else:
                op = os.path.join(args.output_dir, left_file[root_len:])
            left_image = cv2.imread(left_file)
            right_image = cv2.imread(right_file)

            left_copy = left_image.copy()
            if args.model_type == "madnet":
                left_image = preprocess_madnet(left_image)
                right_image = preprocess_madnet(right_image)
            elif args.model_type == "hitnet":
                left_image = preprocess_hit(left_image)
                right_image = preprocess_hit(right_image)

            output = model.forward2((left_image, right_image))

            disp = output[0]

            op = op.replace(".jpg", ".png")

            WriteDepth(disp, left_copy, args.output_dir, op, float(args.bf))

            depth_from_onnx = True

    if args.tof_dir is not None:
        tof_lists = get_files(args.tof_dir)
        if args.tof_selected is not None:
            tof_selected_lists = get_files(args.tof_selected)
            for tof_file, tof_selected_file in zip(tof_lists, tof_selected_lists):
                assert get_last_name(tof_file) == get_last_name(tof_selected_file) \
                    , "tof file and tof select is not same"
        else:
            tof_selected_lists = [None] * len(tof_lists)

    if depth_from_onnx and args.depth_dir:
        assert 0, "depth image from model and depth_dir, please input one"
    elif depth_from_onnx:
        depth_files = get_files(os.path.join(args.output_dir, "depth_psl"))
    elif args.depth_dir is not None:
        depth_files = get_files(args.depth_dir)

    if args.tof_dir is not None and len(tof_lists) > 0:
        root_len = len(args.output_dir)
        for tof_file, tof_selected_file, depth_file in zip(tof_lists, tof_selected_lists, depth_files):
            assert get_last_name(tof_file) == get_last_name(depth_file), "tof file: {} and depth file: {} is not same!".format(tof_file, depth_file)
            print(args.output_dir)
            print(depth_file)
            print(root_len)

            if depth_file[root_len:][0] == '/':
                op = os.path.join(args.output_dir, depth_file[root_len + 1:])
            else:
                op = os.path.join(args.output_dir, depth_file[root_len:])

            compare_depth_tof(args.output_dir, op, depth_file, tof_file, tof_selected_file)


    if args.disp_dir is not None:
        disp_true = get_files(args.disp_dir)
        for disp_file, depth_file in zip(disp_true, depth_file):
            assert get_last_name(disp_file) == get_last_name(
                depth_file), "gt disp file: {} and depth file: {} is not same!".format(tof_file, depth_file)
            compare_depth_disp(args.output_dir, op, depth_file,disp_file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()