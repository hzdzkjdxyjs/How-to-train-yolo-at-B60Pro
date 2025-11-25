# 本章节请在阅读deploy后再使用，（未完成）

## 第1步：下载依赖，版本依赖很烦
```bash
pip install openvino==2024.2.0 openvino-dev==2024.2.0
pip install nncf==2.9.0
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.2/openvino_toolkit_ubuntu20_2024.2.0.15519.tgz -O openvino2024.2.tgz
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

## 第3步：使用量化后的权重推理


