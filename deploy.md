# How-to-deploy-yolo-at-B60Pro
 - How to deploy yolo at B60Pro
 - 接下来的部分是教您如何将模型部署至显卡，需要解释一下的是我们为什么要部署模型，模型训练完后不是可以直接使用的吗？
 - 这是因为纯python方案调用模型或者说原生精度的模型，在GPU上运行的速度是有提升空间的
 - 所以往往采用PT转为onnx通用格式进行部署，例如NV的方案会使用onnx在tensorrt上进行推理，推理的速度会很快
   
---

## 第1步：下载依赖和导出模型为 OpenVINO IR

```bash
pip install openvino openvino-dev
yolo export model=/root/ultralytics/runs/detect/train4/weights/best.pt format=openvino opset=17
````
<img width="1044" height="300" alt="image" src="https://github.com/user-attachments/assets/6c8df529-2a59-4e62-8131-5ff2a6323aba" />

---

## 第2步：编写运行部署代码，创建一个文件夹叫predict.py，先使用原生精度运行运行，时间是14.76ms
 - 写代码如下
```bash
from ultralytics import YOLO
import time

# 加载模型
model = YOLO("/root/ultralytics/runs/detect/train4/weights/best.pt")

# 预热（很重要，OpenVINO 第一次推理会加载图并优化）
model("/root/datasets/coco128/images/train2017/000000000009.jpg",device="xpu")

# 开始计时
start = time.time()
results = model("/root/datasets/coco128/images/train2017/000000000009.jpg")
end = time.time()

print(f"Inference time: {(end - start) * 1000:.2f} ms")
````
 - 运行如下，结果如下

(B60) root@b60:~# /root/anaconda3/envs/B60/bin/python /root/ultralytics/predict.py
Loading /root/ultralytics/runs/detect/train4/weights/best_openvino_model for OpenVINO inference...
Using OpenVINO LATENCY mode for batch=1 inference on (CPU)...

image 1/1 /root/datasets/coco128/images/train2017/000000000009.jpg: 256x256 3 bowls, 1 broccoli, 18.5ms
Speed: 120.7ms preprocess, 18.5ms inference, 922.8ms postprocess per image at shape (1, 3, 256, 256)

image 1/1 /root/datasets/coco128/images/train2017/000000000009.jpg: 256x256 3 bowls, 1 broccoli, 12.8ms
Speed: 0.8ms preprocess, 12.8ms inference, 9.4ms postprocess per image at shape (1, 3, 256, 256)
Inference time: 35.45 ms
 
 - GPU被调用了 
 
<img width="772" height="248" alt="image" src="https://github.com/user-attachments/assets/031d68bc-411c-4086-927c-915fcf22eb5e" />

- 推理大概要40ms
 
<img width="733" height="169" alt="image" src="https://github.com/user-attachments/assets/ae315e8b-a312-472c-8932-78f1acc9fc1e" />

---

## 第3步：使用导出后的模型

```bash
from ultralytics import YOLO
import time

model = YOLO("/root/ultralytics/runs/detect/train4/weights/best_openvino_model")

model.predict("/root/datasets/coco128/images/train2017/000000000009.jpg", device="openvino:GPU.0")

start = time.time()
results = model.predict("/root/datasets/coco128/images/train2017/000000000009.jpg", device="openvino:GPU.0")
end = time.time()

print(f"OpenVINO inference: {(end - start)*1000:.2f} ms")
````

<img width="674" height="212" alt="image" src="https://github.com/user-attachments/assets/4bb89e55-f73f-4a34-b5ed-c0c31730dfbd" />


<img width="779" height="139" alt="image" src="https://github.com/user-attachments/assets/7bd760de-64d5-41d4-aafa-e515ee0ae5a3" />

 - 量化后的模型推理速度是很快的

 - 运行如下，几处说明，为什么显示是用CPU调用的是因为要避免cuda检查，但实际上后台能看到GPU在调用

```bash
(B60) root@b60:~# /root/anaconda3/envs/B60/bin/python /root/ultralytics/predict.py
Loading /root/ultralytics/runs/detect/train4/weights/best_openvino_model for OpenVINO inference...
Using OpenVINO LATENCY mode for batch=1 inference on (CPU)...

image 1/1 /root/datasets/coco128/images/train2017/000000000009.jpg: 256x256 3 bowls, 1 broccoli, 11.4ms
Speed: 2.5ms preprocess, 11.4ms inference, 2.2ms postprocess per image at shape (1, 3, 256, 256)

image 1/1 /root/datasets/coco128/images/train2017/000000000009.jpg: 256x256 3 bowls, 1 broccoli, 2.2ms
Speed: 0.4ms preprocess, 2.2ms inference, 0.7ms postprocess per image at shape (1, 3, 256, 256)
Inference time: 6.01 ms
```

 - 可以看到GPU在运行

<img width="860" height="325" alt="image" src="https://github.com/user-attachments/assets/c66f6e91-52ac-4415-9799-898d8c6e0b7a" />

 - 可以看到相较于没有量化的结果，其速度可以提升3-4倍

<img width="739" height="176" alt="image" src="https://github.com/user-attachments/assets/7d689313-db0c-43f7-911a-25bff43fd9fc" />


