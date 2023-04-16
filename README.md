# results_temp
## yolov5-fire-smoke-detect-python
本仓库保存yolov5训练过程与结果、部分预测结果展示

参考https://github.com/RichardoMrMu/yolov5-fire-smoke-detect-python

我使用了它提供的全部数据集、处理数据集的代码，但直接用它的代码训练会报错（和官方yolov5版本有些不同之处）：RuntimeError: result type Float can‘t be cast to the desired output type long int

我直接使用官方原版 [yolov5](https://github.com/ultralytics/yolov5) 来训练

环境准备
```shell
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

数据集放在datasets/目录，该目录与yolov5/目录同级

在yolov5根目录下，执行以下命令以训练火焰与烟雾检测模型。使用yolov5s预训练模型来fine-tune，数据集信息写在```fire.yaml```。若已有模型不需要训练，则跳过此步骤，直接开始预测

```shell
python train.py --img 640 --epochs 3 --data fire.yaml --weights yolov5s.pt
```

会自动生成runs目录，保存训练过程与结果评价指标图片、最佳模型权重参数、部分验证集预测结果样例图片等。这是本仓库保存的内容

训练过程的报错：多条训练与验证数据报错non-normalized or out of bounds coordinates，这些数据自动被排除，训练与验证过程中不会使用这些数据。影响不是很大

训练完成后，可命令行运行`detect.py`来预测，source指定数据集使用验证集目录下全部图片，weights使用上面训练得到的best.pt。（本仓库没有展示这个预测结果）

```shell
python detect.py --source （要预测的图片路径或其父目录） --weights （模型权重参数weights文件）
如：
python detect.py --source ../datasets/my_fire-smoke/images/val --weights ./runs/train/exp/weights/best.pt
```

也可以用python文件方式来预测（不通过命令行），创建以下python文件在yolov5项目根目录下运行，参考https://github.com/ultralytics/yolov5/issues/36
```python
import torch

# 指定本地yolov5项目路径、custom使用本地模型、模型权重参数文件路径、源代码指定为在本地
model = torch.hub.load('.', 'custom', path='./runs/train/exp/weights/best.pt', source='local')  # or yolov5n - yolov5x6, custom

# 要预测的图片路径
img = "../datasets/my_fire-smoke/images/val/fire_001455.jpg"  

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.show() # 展示预测图片
print(results.pandas().xyxy[0]) # 获取pandas格式预测数据
```


## 训练环境
Google colab免费GPU。将数据集和项目代码上传到Google drive云盘。colab中可以读写drive的数据。
