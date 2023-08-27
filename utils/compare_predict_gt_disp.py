from losses_and_merics import EndPointError, Bad3
import cv2
import tensorflow as tf
import numpy as np

def val_cv2_tf_io(disp_file):

    disp_bytes = tf.io.read_file(disp_file)
    # Using uint16 for higher precision
    disp_map = tf.io.decode_png(disp_bytes, dtype=tf.uint16)
    print("uint16 max: ,min: ",np.max(disp_map), np.min(disp_map))
    disp_map = tf.cast(disp_map, dtype=tf.float32)
    print("float32 max: ,min: ",np.max(disp_map), np.min(disp_map))

    disp_map = disp_map / 256.0

    print("float32/256 max: ,min: ",np.max(disp_map), np.min(disp_map))

    cv_image = cv2.imread(disp_file, -1) /256.0
    print("cv_image max: ,min: ",np.max(cv_image), np.min(cv_image))

    print("sum: {}".format(np.sum(cv_image - disp_map[:,:,0])))

    print(disp_map.shape,cv_image.shape)

def compare_depth_disp(output_dir, op, depth_file,disp_file):
    gt = cv2.imread(disp_file, -1) /256.0



    epe = EndPointError()
    epe.update_state(gt, disparities)
    print("EPE: ", epe.result())
    #
    # bad3 = Bad3()
    # bad3.update_state(gt, disparities)
    # print("Bad3: ", bad3.result())
    #
    # getAbsdiff(disparities, predict_dataset.disp_names_numpy[0])


if __name__ == '__main__':
    compare_depth_disp(1,1,"","/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/test_images/kitti/disp/000000_10.png")