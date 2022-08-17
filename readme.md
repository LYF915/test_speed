## 项目内容
GPU上测试torch/onnx/jax的inference 速度
- torch.profile检测gpu状态
- 实际inference速度


## 项目架构
利用test_speed.py 进行速度测试。主要调用speed package:

- speed
    - jax   (jax inference)
    - onnx  (onnx inference)
    - torch (torch inference)
    - utils (dataload & torch profile)

## 执行
```
bash test_speed.sh
```

## 结果
目前以bert-base测试，目前结果为(单位为 sen/s): 

|   bsz | 32      | 64      | 256      |
|-------|---------|---------|----------|
| torch |  |   | 792.92   |
| jax   |  |   | 812.01   |
|       |  |   | 1.02 x  |

## 环境
```
pip install torch==1.8.0 torchvision==0.8.2+cu92
pip install flax==0.5.3 jax==0.3.15 jaxlib==0.3.15+cuda11.cudnn805 
pip install onnxruntime-gpu==1.8.1 onnx==1.9.0 onnxconverter_common==1.8.1
```


##
https://github.com/google/jax/discussions/8497
