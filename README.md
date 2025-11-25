# How-to-train-yolo-at-B60Pro
 - How to train yolo at B60Pro

 - 这是一个教程，教您如何在B60Pro上运行yolo，yolo会有两部分，训练和推理

---

## 第0步：确定配置
 - 我遇到的很多ISV厂商客户都有反馈说无法运行B60显卡，主要是他们的主板并不支持B60显卡，B60显卡是很特殊的架构，如果是蓝戟的卡那是单芯，如果是铭瑄的卡是很独特的双芯架构，你需要确定你的主板支持x8x8pcie通道分离
 - 当你组装好服务器后，铭瑄的卡你需要在bios处设置开启ResizeBar以及将IIOPCIE启动为auto或x8x8
 - 蓝戟的卡仅需要在bios处设置开启ResizeBar即可，蓝戟的卡不需要确定支不支持x8x8因为他不需要通道分离

 Enable Re-Size BAR Support and PCIe Gen5 X8X8 as below:
<img width="1158" height="632" alt="image" src="https://github.com/user-attachments/assets/ea594ad5-a698-45d3-8998-ff2be6e983ea" />

<img width="1086" height="650" alt="image" src="https://github.com/user-attachments/assets/d764d296-ea73-4dc7-b046-75b20380ea87" />

<img width="1173" height="729" alt="image" src="https://github.com/user-attachments/assets/762aa6a7-9b31-4462-88b3-b89f0130b03b" />

<img width="1271" height="846" alt="image" src="https://github.com/user-attachments/assets/6beff878-e51f-4b47-9746-7ad7856e1efa" />

<img width="1044" height="705" alt="image" src="https://github.com/user-attachments/assets/0a761e9a-3c81-45d2-9d79-e1c9a5e41e03" />

<img width="1169" height="754" alt="image" src="https://github.com/user-attachments/assets/7b60018e-1dd8-4237-857b-a1dbb6f075c7" />


---

## 第一步：安装系统和基础环境配置环境

 - 您需要在服务器安装ubuntu系统25.04，并确认您的服务器版本的内核是25.04的原生内核
 - 接下来您需要安装驱动，安装驱动教程可以参考这个教程中的1.1部分Install Bare Metal Environment：https://github.com/intel/llm-scaler/blob/main/vllm/README.md/#1-getting-started-and-usagexit
 - 还要提到一点的是，b60的xpu-smi的监控不设置的话只支持root权限，需要设置
```bash
sudo gpasswd -a ${USER} render
sudo newgrp render
sudo reboot
````

---

## 第二步：正式安装
 - 您需要安装anaconda环境并安装相关依赖
 - 国内环境请设置国内pip镜像
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
````
 - 下载项目
```bash
git clone https://github.com/ultralytics/ultralytics.git
````
 - 创建环境
```bash
conda create -n b60 python=3.10 -y
conda activate b60
````
 - 安装依赖
```bash
cd ultralytics
pip install -e .
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
````
 - 检查torch安装情况，和我一样就是安装完成了
```bash
(B60) root@b60:~/ultralytics# python
Python 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.version.xpu)
20250101
>>> print(torch.xpu.is_available())
True
>>> print(torch.xpu.get_device_name(0))
Intel(R) Graphics [0xe211]
````

---

## 第二步：正式训练

 - 修改源码使其支持xpu训练，找到ultralytics/ultralytics/utils/torch_utils.py文件select_device函数，在注释处添加这个
```bash
    if str(device).startswith(("xpu", "intel")):
    # If PyTorch has XPU and the device is available
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            if ":" in device:
                # xpu:0 → index 0
                index = int(device.split(":")[1])
                return torch.device(f"xpu:{index}")
            else:
                return torch.device("xpu")
        else:
            raise ValueError(
                f"Intel XPU device requested but not available.\n"
                f"torch.version.xpu: {torch.version.xpu}\n"
                f"torch.xpu.is_available(): {torch.xpu.is_available()}\n"
            )
    if isinstance(device, torch.device) or str(device).startswith(("tpu", "intel")):
        return device
````
 - 找到/ultralytics/ultralytics/utils/checks.py 修改amp检查函数，直接替换
```bash
def check_amp(model):
    """
    Safe AMP check for CUDA / XPU.
    - For Intel XPU: always disable AMP (not supported by IPEX)
    - For CPU/MPS: AMP disabled
    - For CUDA GPUs: run original AMP capability checks
    """

    import torch
    import re
    from ultralytics.utils.torch_utils import autocast

    device = next(model.parameters()).device
    device_type = device.type.lower()
    prefix = "AMP: "

    # --------------------------------------------------
    # 1. Intel XPU → AMP 不支持，必须关闭
    # --------------------------------------------------
    if hasattr(torch, "xpu") and torch.xpu.is_available() and device_type == "xpu":
        LOGGER.warning(f"{prefix}Intel XPU detected. AMP is disabled (not supported on XPU).")
        return False

    # --------------------------------------------------
    # 2. CPU 或 Apple MPS → 直接关闭 AMP
    # --------------------------------------------------
    if device_type in {"cpu", "mps"}:
        return False

    # --------------------------------------------------
    # 3. CUDA GPU 才进行 AMP 检查
    # --------------------------------------------------
    if device_type == "cuda":
        try:
            gpu = torch.cuda.get_device_name(device)
        except Exception:
            # CUDA 不可用或无效
            return False

        # 低端/问题 GPU 列表
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)",
            re.IGNORECASE,
        )

        # 如果是低端 CUDA，就警告
        if bool(pattern.search(gpu)):
            LOGGER.warning(
                f"{prefix}checks failed ❌. AMP on {gpu} may cause NaN losses. AMP disabled."
            )
            return False

        # 对于其它 CUDA GPU → 默认启用 AMP
        return True

    # --------------------------------------------------
    # 4. 其它未知设备（例如 OpenCL / HIP） → 禁用 AMP
    # --------------------------------------------------
    return False
````

 - 找到ultralytics/engine/trainer.py文件，修改函数_get_memory和_clear_memory函数


```bash
    def _clear_memory(self, threshold=0.5):
        """
        Memory cleaner for CUDA / XPU / CPU.
        Prevents VRAM spikes. On XPU, try clear_cached_memory.
        """

        device = self.device
        device_type = device.type if isinstance(device, torch.device) else str(device)

        # XPU cleanup
        if device_type == "xpu":
            try:
                if self._get_memory(fraction=True) <= threshold:
                    # XPU 清理缓存（IPEX >= 2.1 支持）
                    if hasattr(torch.xpu, "empty_cache"):
                        torch.xpu.empty_cache()
            except Exception:
                pass
            return

        # CUDA cleanup
        if device_type == "cuda":
            try:
                if self._get_memory(fraction=True) <= threshold:
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return

        # CPU: no action
        return

    def _get_memory(self, fraction=False):
        """
        Universal memory monitor for CUDA / XPU / CPU.
        Returns memory usage in bytes or fraction.
        """
        device = self.device
        device_type = device.type if isinstance(device, torch.device) else str(device)
        # ---------------------------
        # Intel XPU version
        # ---------------------------
        if device_type == "xpu":
            try:
                props = torch.xpu.get_device_properties(device)
                total = props.total_memory
                reserved = torch.xpu.memory_reserved(device)
                allocated = torch.xpu.memory_allocated(device)
                # fallback for older IPEX that lacks memory_allocated
                allocated = allocated if allocated is not None else 0
                if fraction:
                    return allocated / total if total > 0 else 0
                return allocated
            except Exception:
                return 0.0
        # ---------------------------
        # CPU (no VRAM)
        # ---------------------------
        if device_type == "cpu":
            return 0.0
        # ---------------------------
        # CUDA path (original)
        # ---------------------------
        try:
            total = torch.cuda.get_device_properties(device).total_memory
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            return (allocated / total) if fraction else allocated
        except Exception:
            return 0.0
````

## 第三步：正式训练
```bash
yolo train model=yolov8n.pt data=coco128.yaml epochs=3 imgsz=256 device=xpu
````
<img width="1031" height="1044" alt="image" src="https://github.com/user-attachments/assets/a260ec76-26c9-4485-9eae-8d982fd6d2da" />


















