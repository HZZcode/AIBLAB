import pandas as pd
import numpy as np
from numpy.typing import NDArray as ND
import torch
from torch.utils import data
from torch import nn

# 进行拟合系数的函数
def fit(x:pd.DataFrame, y:ND, device):
    '''
    系数拟合(只考虑一维情况)
    '''
    # 将pd数据转为np数组
    X = torch.tensor(x.to_numpy(), device=device)
    y = torch.tensor(y, device=device)

    # 检查数据形状，若不是列向量则转为列向量
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # 创造储存结果的数组
    r2_out = [None] * (X.shape[1])
    p_out = [None] * (X.shape[1])
    loss_out = [None] * (X.shape[1])

    # X的每列数据即为y=a*f(x1,x2,..)+b中的f(x1,x2,..)方程数据
    # 对每列数据分别拟合系数
    for i in range(X.shape[1]):
        # 解析解
        r2, p, loss = analytical_solving(X[:, i], y[:])
        # 神经网络
        # r2, p, loss = net_solving(X[:, i], y[:], device)
        
        # 保存结果
        r2_out[i] = r2
        p_out[i] = p
        loss_out[i] = loss

    return r2_out, p_out, loss_out


def get_loss(y_true, y_pred):
    '''
    计算损失函数
    '''
    return float(torch.mean((y_true - y_pred) ** 2) / 2)

def get_r2(y_true, y_pred):
    '''
    计算R2
    '''
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)  # 总平方和
    ss_residual = torch.sum((y_true - y_pred) ** 2)   # 残差平方和
    return float(1 - (ss_residual / ss_total))

def analytical_solving(X:torch.tensor, y:torch.tensor, lambda_reg=1e-6):
    '''
    lambda_reg: 正则化系数（防止 X^T X 不可逆）
    '''
    # 检查数据形状，若不是列向量则转为列向量
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # 增加一列全为1的列，用于计算截距
    ones = torch.ones((X.shape[0], 1), device=X.device)
    X = torch.hstack((X, ones))

    # 计算解析解
    X_T_X = X.T @ X
    X_T_X_reg = X_T_X + lambda_reg * torch.eye(X_T_X.shape[0], dtype=torch.float64, device=X.device)
    X_T_y = X.T @ y
    XTX_inv = torch.linalg.inv(X_T_X_reg)
    p = XTX_inv @ X_T_y

    r2 = get_r2(y, X @ p)
    loss = get_loss(y, X @ p)
    
    # 展平p
    p = p.flatten().tolist()

    return r2, p, loss

def net_solving(X:torch.tensor, y:torch.tensor, device, num_epochs=2000, batch_size=32, lr=0.01):
    def load_array(data_arrays, batch_size, is_train=False):
        """
        构造一个PyTorch数据迭代器。
        """
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    features = X.float()
    labels = y.float()

    # 创建数据迭代器
    data_iter = load_array((features, labels), batch_size)

    # 定义模型
    net = nn.Sequential(nn.Linear(1, 1)).to(device)

    # 初始化模型参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # 定义损失函数和优化器
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 训练模型
    for epoch in range(num_epochs):
        for X_iter, y_iter in data_iter:
            l = loss(net(X_iter) ,y_iter)
            trainer.zero_grad()
            l.backward()
            trainer.step()

    # 获取模型参数
    p = np.array([])
    for w in net[0].weight.data.cpu().numpy():
        p = np.append(p, w)

    p = np.append(p, net[0].bias.data.cpu().numpy()[0])
    p = p.reshape(-1, 1)

    # 计算R2
    # X = np.hstack((X, np.ones((X.shape[0], 1))))
    # r2 = get_r2(y, X @ p)
    p_tensor = torch.tensor(p, dtype=torch.float32, device=device)
    ones = torch.ones((X.shape[0], 1), device=device)
    X_with_bias = torch.hstack((X.float(), ones))
    r2 = get_r2(y.float(), X_with_bias @ p_tensor)

    # 展平p
    p = [item for sublist in p for item in sublist]

    return r2, p, float(loss(net(features), labels))
