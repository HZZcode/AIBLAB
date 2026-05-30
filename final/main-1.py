from expansion import expand
from result_sorting import sort_result
from result_displaying import display_result
from sis_ana import sis
from coefficients_fitting import fit
from result_plotting import plot_result
import pandas as pd
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 代码运行目录
path = os.getcwd()
# path = "yourfilepath"

# 读取数据
data = pd.read_csv(os.path.join(path, "data.csv"))
focus = pd.read_csv(os.path.join(path, "focus.csv"))

# 初始输入的扩充
data_expanded = expand(data)

# 确定性独立筛选
data_sis = sis(data_expanded, focus.to_numpy(), 10)
# data_sis = data

# 系数拟合
r2, coef, loss = fit(data_sis, focus.to_numpy(), device)

# 结果整理
results = sort_result(data_sis.to_numpy(), 
                      data_sis.columns.to_numpy(),
                      r2, coef, loss, 10)

# 输出日志
display_result(results, 10, path)

# 绘制图像
plot_result(results[0], focus, path)
