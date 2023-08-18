

## test example demo
### 1. only output depth(disp) image
```angular2html
python main.py --data_dir test_images --img_height 400 --img_width 640 --output_dir result --onnx_file *.onnx --bf 3424
```
input ```--data_dir``` is a dir containing left(cam0) and right(cam0) dirs for test image  
output 
    dir ```result/color、result/depth、result/gray``` is disp result for show
    dir ```depth_psl``` is used for tof calculation
