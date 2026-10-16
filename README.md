# Time Series Prediction

时间序列预测研究项目：通胀与澳大利亚月度啤酒产量。

## 当前状态

已完成目录搭建、数据整理和部分已有代码迁移，**不代表全部实验已经实现或验证**。
原桌面 prediction 中的文件保持不变；旧笔记本及输出仍在那里可查。

| 入口 | 状态 |
| --- | --- |
| inflation/01_baseline | 可运行的训练均值基线，按时间前90%/后10%划分 |
| inflation/02_fourier | Fourier ratio 诊断；预测待实现 |
| inflation/03_trigonometric | 随机三角重构示例；不属于样本外预测 |
| inflation/04_lstm | 入口已建立；训练待核查 |
| inflation/05_stl | 入口已建立；待确定通胀频率和周期 |
| inflation/06_missing_data | 入口已建立；原 L1 实验待迁移 |
| beer/01_stl_decomposition | 月度 period=12 分解 |
| beer/02_component_forecasting | 入口已建立；分量预测待实现 |

## 使用方法

Python 3.10 或以上。在 VS Code 中用 Open Folder 打开整个项目，选择 Python/Jupyter 扩展及合适的 Python kernel。
在项目根目录按需安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pytest tests
```

打开 experiments 中的笔记本，从上到下运行。开头会从当前目录向上定位项目；数据和结果路径均基于项目根目录。
requirements.txt 尚未锁定版本，不是已验证的完整环境快照。

## 目录

- data：实验输入数据。
- src/preprocessing：数据清洗。
- src/models：可重复使用的模型和诊断函数。
- src/evaluation：指标函数。
- experiments：分数据集的实验入口；编号表示组织顺序，各笔记本独立加载数据。
- results：图表与指标；.gitkeep 让空目录能被 Git 保存。
- tests：清洗及数据迁移检查。

## 数据来源与整理

- data/inflation.csv：由原 prediction/inflation.csv 的 date(90%)/inflation rate 和 date(10%)/data 两组列拼接为 ds,y，去除无效日期/数值，按日期排序。保留 1252 行，范围 2020-02-07 至 2025-02-07。旧预测和误差列未带入。原始文件仍保留在桌面。
- data/australia_beer.csv：原 Imputation codes (1)/Beer.csv 原样复制，仅修改文件名；保留 ds,y,index 列。
- 两个数据集的外部来源和分享许可仍需补充。

## 代码来源与未完成事项

- cleaning_pipeline.py 来自原 Imputation codes (1)/src，逻辑未改。
- lstm.py 来自 LSTM-code.ipynb（此前由 LSTM-code.txt 转换）；baseline.py 和 metrics.py 提取其中对应函数。
- 原 LSTM 的测试目标提前截断、训练 loss 张量维度以及 CVRMSE 公式待核查。本次不宣称训练结果正确。
- trigonometric.py 提取 trig poly.txt 函数；将 eval_trig_poly 的采样长度从 N//12 改为 N，使其与输出数组长度一致。未定义变量 z 的旧实验段没有迁入。
- fourier.py 提取 inflation trig.ipynb 中的 Fourier ratio 计算；原笔记本其他拟合和调参流程待迁移。
- stl.py 封装 STL-code.txt 中的分解调用。
- 原 test_cleaning_pipeline.py 是依赖旧文件名的手动检查脚本。此处改为 pytest 测试，匹配新数据路径。
- 原 L1minimization+.ipynb 和本地 lstm_on_stl_trend.py 保留原位置，未迁移为完成的实验。

## GitHub

本次没有初始化或上传仓库。先检查数据及实验，再在 VS Code 初始化 Git、提交并发布。

## 本次检查

Python 源码语法与 notebook 格式已检查。两个清洗测试通过。当前 Anaconda 的 pytest 默认调试插件加载 readline 时崩溃；使用 `python -m pytest -p no:debugging tests -q` 可运行通过。当前检查环境未安装 torch，因此未验证 LSTM 训练。
