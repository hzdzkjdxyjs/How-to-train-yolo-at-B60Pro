# 本章节请在阅读deploy后再使用，（未完成）
## 第1步：下载依赖，版本依赖很烦
```bash
pip install onnx
````

---

## 第2步：将模型转为onnx格式
```bash
yolo export model=/root/ultralytics/runs/detect/train4/weights/best.pt format=onnx opset=12
````
---

## 第3步：新建文件/root/ultralytics/quantize.py
```bash
import glob
import numpy as np
from PIL import Image
from openvino.runtime import Core
import torch
from nncf import Dataset, quantize
from openvino.runtime import serialize
import os

# --- 0. 定义量化输出目录（保持不变） ---
save_dir = "/root/ultralytics/runs/detect/train4/weights/best_openvino_model"
os.makedirs(save_dir, exist_ok=True)

# --- 1. 加载原始 ONNX 模型（你的要求：保持原来的路径） ---
onnx_path = "/root/ultralytics/runs/detect/train4/weights/best.onnx"
ie = Core()
model = ie.read_model(onnx_path)

# --- 2. 定义校准数据预处理函数 ---
def preprocess_img(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img).transpose(2,0,1)[None].astype(np.float32)
    return arr

# --- 3. 构建 NNCF 校准 Dataset ---
image_paths = glob.glob("/root/datasets/coco128/images/train2017/*.jpg")
calib_dataset = Dataset(image_paths, preprocess_img)

# --- 4. 执行量化（INT8 模式） ---
quant_model = quantize(model, calib_dataset)

# --- 5. 保存量化后的模型：放到 best_openvino_model 目录 ---
xml_path = f"{save_dir}/best_openvino_model_int8/best_openvino_int8.xml"
bin_path = f"{save_dir}/best_openvino_model_int8/best_openvino_int8.bin"

serialize(quant_model, xml_path, bin_path)
print(f"量化完成：{xml_path} / {bin_path}")

# --- 6. 推理测试（可选） ---
compiled = ie.compile_model(ie.read_model(xml_path), "GPU")
dummy = np.zeros((1,3,256,256), dtype=np.float32)
res = compiled([dummy])
print("推理结果 shape:", res[0].shape)
````
---
<img width="1332" height="86" alt="image" src="https://github.com/user-attachments/assets/d52fd994-c9d9-496f-94d3-9d57d1f4730d" />


## 第4步：推理
```bash
(B60) root@b60:~# yolo val model=/root/ultralytics/runs/detect/train4/weights/best_openvino_model_int8 data=coco128.yaml device=openvino:GPU.0 imgsz=256
Loading /root/ultralytics/runs/detect/train4/weights/best_openvino_model_int8 for OpenVINO inference...
Using OpenVINO LATENCY mode for batch=1 inference on (CPU)...
Setting batch=1 input of shape (1, 3, 256, 256)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2156.8±1013.5 MB/s, size: 44.2 KB)
val: Scanning /root/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 128/128 1.5Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 128/128 177.3it/s 0.7s
                   all        128        929    0.00102      0.019   0.000906   0.000444
                person         61        254          0          0          0          0
               bicycle          3          6          0          0          0          0
                   car         12         46          0          0          0          0
            motorcycle          4          5          0          0          0          0
              airplane          5          6          0          0          0          0
                   bus          5          7          0          0          0          0
                 train          3          3          0          0          0          0
                 truck          5         12          0          0          0          0
                  boat          2          6          0          0          0          0
         traffic light          4         14          0          0          0          0
             stop sign          2          2          0          0          0          0
                 bench          5          9          0          0          0          0
                  bird          2         16          0          0          0          0
                   cat          4          4          0          0          0          0
                   dog          9          9     0.0137      0.111     0.0175      0.014
                 horse          1          2          0          0          0          0
              elephant          4         17          0          0          0          0
                  bear          1          1          0          0          0          0
                 zebra          2          4          0          0          0          0
               giraffe          4          9          0          0          0          0
              backpack          4          6          0          0          0          0
              umbrella          4         18          0          0          0          0
               handbag          9         19          0          0          0          0
                   tie          6          7          0          0          0          0
              suitcase          2          4          0          0          0          0
               frisbee          5          5          0          0          0          0
                  skis          1          1          0          0          0          0
             snowboard          2          7          0          0          0          0
           sports ball          6          6          0          0          0          0
                  kite          2         10          0          0          0          0
          baseball bat          4          4          0          0          0          0
        baseball glove          4          7          0          0          0          0
            skateboard          3          5          0          0          0          0
         tennis racket          5          7          0          0          0          0
                bottle          6         18          0          0          0          0
            wine glass          5         16          0          0          0          0
                   cup         10         36          0          0          0          0
                  fork          6          6          0          0          0          0
                 knife          7         16          0          0          0          0
                 spoon          5         22          0          0          0          0
                  bowl          9         28          0          0          0          0
                banana          1          1          0          0          0          0
              sandwich          2          2          0          0          0          0
                orange          1          4          0          0          0          0
              broccoli          4         11          0          0          0          0
                carrot          3         24          0          0          0          0
               hot dog          1          2          0          0          0          0
                 pizza          5          5     0.0238        0.6     0.0232    0.00744
                 donut          2         14          0          0          0          0
                  cake          4          4          0          0          0          0
                 chair          9         35          0          0          0          0
                 couch          5          6     0.0155      0.333     0.0107    0.00368
          potted plant          9         14   0.000325     0.0714   0.000177   7.09e-05
                   bed          3          3          0          0          0          0
          dining table         10         13     0.0189      0.231     0.0127    0.00632
                toilet          2          2          0          0          0          0
                    tv          2          2          0          0          0          0
                laptop          2          3          0          0          0          0
                 mouse          2          2          0          0          0          0
                remote          5          8          0          0          0          0
            cell phone          5          8          0          0          0          0
             microwave          3          3          0          0          0          0
                  oven          5          5          0          0          0          0
                  sink          4          6          0          0          0          0
          refrigerator          5          5          0          0          0          0
                  book          6         29          0          0          0          0
                 clock          8          9          0          0          0          0
                  vase          2          2          0          0          0          0
              scissors          1          1          0          0          0          0
            teddy bear          6         21          0          0          0          0
            toothbrush          2          5          0          0          0          0
Speed: 0.1ms preprocess, 2.6ms inference, 0.0ms loss, 0.4ms postprocess per image
Results saved to /root/ultralytics/runs/detect/val14
💡 Learn more at https://docs.ultralytics.com/modes/val
VS Code: view Ultralytics VS Code Extension ⚡ at https://docs.ultralytics.com/integrations/vscode
````
 # 我也不知道为什么效果那么差，非常好奇原因？
 
 # 另外一种思路 

## 第1步：下载依赖，版本依赖很烦
```bash
pip install openvino==2024.2.0 openvino-dev==2024.2.0
pip install nncf==2.9.0
````

## 第2步：新建文件/root/ultralytics/quantize.py
```bash
import glob
import numpy as np
from PIL import Image
from openvino.runtime import Core
import torch
from nncf import Dataset, quantize
from openvino.runtime import serialize
# --- 1. 加载 OpenVINO 模型 ---
ie = Core()
model = ie.read_model("/root/ultralytics/runs/detect/train4/weights/best_openvino_model/best.xml")
# --- 2. 定义校准数据预处理函数 ---
def preprocess_img(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img).transpose(2,0,1)[None].astype(np.float32)
    return arr
# --- 3. 构建 NNCF 校准 Dataset ---
image_paths = glob.glob("/root/datasets/coco128/images/train2017/*.jpg")
calib_dataset = Dataset(image_paths, preprocess_img)
# --- 4. 执行量化（INT8 模式） ---
quant_model = quantize(model, calib_dataset)
# --- 5. 保存量化后的模型 ---
serialize(quant_model, "best_openvino_int8.xml", "best_openvino_int8.bin")
print("量化完成：best_openvino_int8.xml / best_openvino_int8.bin")
# --- 6. 推理测试（可选） ---
compiled = ie.compile_model(ie.read_model("best_openvino_int8.xml"), "GPU")
dummy = np.zeros((1,3,256,256), dtype=np.float32)
res = compiled([dummy])
print("推理结果 shape:", res[0].shape)
````
---
<img width="1332" height="86" alt="image" src="https://github.com/user-attachments/assets/1c20a4f8-8589-412c-ac55-54e69d575bf3" />


## 第3步：使用量化后的权重推理
```bash
(B60) root@b60:~# yolo val model=/root/ultralytics/runs/detect/train4/weights/best_openvino_model_int8 data=coco128.yaml device=openvino:GPU.0 imgsz=256
Loading /root/ultralytics/runs/detect/train4/weights/best_openvino_model_int8 for OpenVINO inference...
Using OpenVINO LATENCY mode for batch=1 inference on (CPU)...
Setting batch=1 input of shape (1, 3, 256, 256)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2340.7±1298.5 MB/s, size: 64.1 KB)
val: Scanning /root/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 128/128 1.8Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 128/128 187.1it/s 0.7s
                   all        128        929    0.00118     0.0105   0.000872   0.000291
                person         61        254          0          0          0          0
               bicycle          3          6          0          0          0          0
                   car         12         46          0          0          0          0
            motorcycle          4          5          0          0          0          0
              airplane          5          6          0          0          0          0
                   bus          5          7          0          0          0          0
                 train          3          3          0          0          0          0
                 truck          5         12          0          0          0          0
                  boat          2          6          0          0          0          0
         traffic light          4         14          0          0          0          0
             stop sign          2          2          0          0          0          0
                 bench          5          9    0.00585      0.111    0.00347   0.000694
                  bird          2         16          0          0          0          0
                   cat          4          4          0          0          0          0
                   dog          9          9          0          0          0          0
                 horse          1          2          0          0          0          0
              elephant          4         17          0          0          0          0
                  bear          1          1          0          0          0          0
                 zebra          2          4          0          0          0          0
               giraffe          4          9          0          0          0          0
              backpack          4          6          0          0          0          0
              umbrella          4         18          0          0          0          0
               handbag          9         19          0          0          0          0
                   tie          6          7          0          0          0          0
              suitcase          2          4          0          0          0          0
               frisbee          5          5          0          0          0          0
                  skis          1          1          0          0          0          0
             snowboard          2          7          0          0          0          0
           sports ball          6          6          0          0          0          0
                  kite          2         10          0          0          0          0
          baseball bat          4          4          0          0          0          0
        baseball glove          4          7          0          0          0          0
            skateboard          3          5          0          0          0          0
         tennis racket          5          7          0          0          0          0
                bottle          6         18          0          0          0          0
            wine glass          5         16          0          0          0          0
                   cup         10         36          0          0          0          0
                  fork          6          6          0          0          0          0
                 knife          7         16          0          0          0          0
                 spoon          5         22          0          0          0          0
                  bowl          9         28     0.0175     0.0714     0.0102    0.00355
                banana          1          1          0          0          0          0
              sandwich          2          2          0          0          0          0
                orange          1          4          0          0          0          0
              broccoli          4         11          0          0          0          0
                carrot          3         24          0          0          0          0
               hot dog          1          2          0          0          0          0
                 pizza          5          5          0          0          0          0
                 donut          2         14          0          0          0          0
                  cake          4          4          0          0          0          0
                 chair          9         35          0          0          0          0
                 couch          5          6          0          0          0          0
          potted plant          9         14          0          0          0          0
                   bed          3          3     0.0435      0.333     0.0368      0.011
          dining table         10         13     0.0169      0.231     0.0116    0.00538
                toilet          2          2          0          0          0          0
                    tv          2          2          0          0          0          0
                laptop          2          3          0          0          0          0
                 mouse          2          2          0          0          0          0
                remote          5          8          0          0          0          0
            cell phone          5          8          0          0          0          0
             microwave          3          3          0          0          0          0
                  oven          5          5          0          0          0          0
                  sink          4          6          0          0          0          0
          refrigerator          5          5          0          0          0          0
                  book          6         29          0          0          0          0
                 clock          8          9          0          0          0          0
                  vase          2          2          0          0          0          0
              scissors          1          1          0          0          0          0
            teddy bear          6         21          0          0          0          0
            toothbrush          2          5          0          0          0          0
Speed: 0.1ms preprocess, 1.9ms inference, 0.0ms loss, 0.4ms postprocess per image
Results saved to /root/ultralytics/runs/detect/val16
💡 Learn more at https://docs.ultralytics.com/modes/val
VS Code: view Ultralytics VS Code Extension ⚡ at https://docs.ultralytics.com/integrations/vscode
````
