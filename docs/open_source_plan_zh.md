# FuDU 开源规划

这份开源版本定位为 **FuDU 官方研究代码**，目标是让读者可以复现 FuDU 的主动学习采样逻辑，并方便接入不同检测器工程。

## 1. 开源范围

本仓库开源以下内容：

- PGUQ：基于 normal/defect 原型的图像级全局不确定性计算。
- DeUE：基于分类熵、定位熵和置信度的框级缺陷不确定性计算。
- FuDU fuzzy sampler：将 `Ug` 和 `Ud` 映射到 `DNS/LS/HS/MS` 采样动作的模糊推理规则。
- Stream scoring CLI：对连续输入图像打分，并导出待标注样本列表。
- Optional PyTorch helpers：用于把 learnable prototypes 接入自定义 PyTorch detector 的轻量模块。
- Synthetic example：不含真实工业数据的示例，便于验证安装和流程。
- 论文方法图：仅用于说明框架和采样逻辑，不代表公开底层工业数据。

## 2. 不开源范围

以下内容不包含在仓库中：

- 真实核燃料棒缺陷数据集。
- 任何公开数据集的重新分发副本。
- 训练好的检测器权重或中间 checkpoint。
- 本地实验输出、日志、可视化结果、`runs/` 目录。
- 涉及企业/实验室路径的配置文件。
- 可反向还原完整数据集的原始样本集合。

`.gitignore` 已屏蔽 `data/`、`datasets/`、`weights/`、`checkpoints/`、`*.pt`、`*.pth`、`*.npz`、`runs/` 等文件。

## 3. 用户复现路径

推荐用户按下面方式接入自己的检测器：

1. 使用私有 labeled subset 训练一个 initial detector。
2. 从 detector backbone 或 neck 提取每张图像的 image-level feature。
3. 用 `fudu build-prototypes` 构建 normal/defect prototype library。
4. 用 detector 对每个 stream batch 推理，导出分类概率、定位概率或归一化熵。
5. 用 `fudu score` 计算 `Ug`、`Ud`、模糊动作和采样概率。
6. 标注 `selected=1` 的图像。
7. 将新标注样本加入训练集，更新 detector 和 prototype library，进入下一轮主动学习。

## 4. 项目结构

```text
FuDU/
  src/fudu/
    prototypes.py      PGUQ 原型库与 NumPy K-means
    uncertainty.py     DeUE 双熵不确定性
    fuzzy.py           模糊隶属函数、规则库和采样概率
    stream.py          流式打分 pipeline
    cli.py             命令行入口
    torch_modules.py   可选 PyTorch 集成模块
  examples/            合成示例
  docs/                数据格式、开源规划和集成说明
  tests/               单元测试
```

## 5. GitHub 上传前检查清单

- 修改 `README.md` 中的 `https://github.com/<your-org>/FuDU`。
- 修改 `CITATION.cff` 中的 `repository-code`。
- 如学校/课题组对开源协议有要求，替换 `LICENSE`。
- 删除本地生成的 `runs/` 目录后再上传，或确认 `.gitignore` 生效。
- 用下面命令确认项目能跑：

```bash
python examples/synthetic_stream.py
python -m unittest discover -s tests
```

## 6. 后续扩展建议

第一版建议保持轻量。后续可以逐步增加：

- 针对 RT-DETR/DEIM/MR-DETR/YOLO 的 feature/prediction adapter。
- 完整 active-learning round runner。
- 可视化 sampling surface 和 batch-level sampling report。
- 基于验证集的 fuzzy membership 自动校准脚本。
- 与 Ultralytics/Detectron2/MMDetection 的训练命令适配示例。
