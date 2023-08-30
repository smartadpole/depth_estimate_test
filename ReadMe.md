
## Environment
1. pip install -r requirements.txt
2. confirm file ```/usr/share/fonts/Fonts/simsun.ttf``` is exists.
## new test example
### KITTI
 - no disp
```angular2html
python main.py --data_dir /home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/test_images/kitti/temp/ --img_height 375 --img_width 1424 --onnx_file /home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/D10.2.13_ --output_dir ./result/D10.2.13/kitti_2000_no
```
 - with disp
```angular2html
python main.py --data_dir /home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/test_images/kitti_train/REMAP --img_height 375 --img_width 1424 --onnx_file /home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/D10.2.14_375_1242/epoch-1700.onnx --disp_dir /home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/test_images/kitti_train/disp --output_dir ./result/D10.2.14/kitti_1700
```
### Parker
```angular2html
python main.py --data_dir /data/Parker/REMAP/TEST/pick_mask_REMAP  --disp_dir /data/Parker/test_result/NPU/depth_data/pick_mask/result_tof --img_height 360 --img_width 576 --onnx_file /home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/D10.2.13_576_360/epoch-2000.onnx --output_dir ttt1 --bf 3424 --center_crop 0.9
```
### Rubby
```angular2html
 python main.py --data_dir /home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/test_images/rubby --img_height 360 --img_width 576 --onnx_file /home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/D10.2.14_576_360/epoch-1700.onnx   --bf 1420 --center_crop 0.9 --output_dir ./result/D10.2.14/rubby-1700
```
## old version test example demo
### 1. only output depth(disp) images
```angular2html
python main.py --data_dir test_images --img_height 400 --img_width 640 --output_dir result --onnx_file *.onnx --bf 3424
```
input ```--data_dir``` is a dir containing left(cam0) and right(cam0) dirs for test image  
output 
    dir ```result/color、result/depth、result/gray``` is disp result for show
    dir ```depth_psl``` is used for tof calculation

### 2.output depth(disp) images, and compare it with tof and tof with selected
```angular2html
python main.py --data_dir image_dir --img_height 400 --img_width 640 --output_dir result_parker --onnx_file *.onnx --bf 3424 --tof_dir result_tof --tof_selected result_image_with_tof
```
```--data_dir``` 、```--tof_dir ``` 、 ```--tof_selected``` in cam0 should contain same images, if all selected point's tof value * depth == zero, continue next image.  
output  
    dir ```result/color、result/depth、result/gray``` is disp result for show
    dir ```depth_psl``` is used for tof calculation
    dir ```tof_error``` contain result
### 3. output kitti disp diff
```angular2html
python main.py --datra_dir /home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/test
_images/kitti_train/REMAP --img_height 375  --img_width 1242 --disp_dir /home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/test_images/kitti_train/disp --onnx_file /home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/kitti.onnx/kitti.onnx
```