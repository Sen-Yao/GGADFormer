# GGADFormer

## Requirements

- CUDA 11.8

To install requirements:

```bash
conda env create -f environment.yml
conda activate GGADFormer
```

## 半监督图异常检测

半监督图异常检测任务中，异常节点占极少数（10%以内）。训练时模型只能获取一部分的正常节点及标签，任何异常节点及其标签都无法访问到。