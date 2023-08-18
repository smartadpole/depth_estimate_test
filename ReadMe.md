
## Environment
pip install -r requirements.txt
## test example demo
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
--data_dir image_dir --img_height 400 --img_width 640 --output_dir result_parker --onnx_file *.onnx --bf 3424 --tof_dir result_tof --tof_selected result_image_with_tof
```
```--data_dir``` 、```--tof_dir ``` 、 ```--tof_selected``` in cam0 should contain same images 
output  
    dir ```result/color、result/depth、result/gray``` is disp result for show
    dir ```depth_psl``` is used for tof calculation
    dir ```tof_error``` contain result
  