import pandas as pd
import numpy as np
import itertools
from numpy.typing import NDArray as ND

def expand(ori_data: pd.DataFrame):
    '''
    对数据进行扩展 (7.6.3 节重构版)
    '''
    ## 初始化
    out_data = ori_data.copy()
    out_num = ori_data.to_numpy()
    out_str = ori_data.columns.to_numpy()

    ## 变换：[x1, x2, x3, ...] -> [[x1], [x2], [x3],...]
    out_num = np.hsplit(out_num, out_num.shape[1])
    out_str = np.hsplit(out_str, out_str.shape[0])

    ## 1. 一元特征扩展（幂运算 + 高级算子）
    for index, name in enumerate(out_str):
        # 原有的幂运算
        num_p1, str_p1 = power(out_num[index], name, 1/2)
        num_p2, str_p2 = power(out_num[index], name, 2)
        num_p3, str_p3 = power(out_num[index], name, 1/3)
        num_p4, str_p4 = power(out_num[index], name, 3)
        
        # 7.6.3 新增：高级一元算子（倒数、指数、对数）
        num_adv, str_adv = inv_and_exp_log(out_num[index][:, :1], name)
        
        # 合并所有一元扩展特征
        out_num[index] = np.hstack((out_num[index], num_p1, num_p2, num_p3, num_p4, num_adv))
        out_str[index] = np.hstack((out_str[index], str_p1, str_p2, str_p3, str_p4, str_adv))

    ## 关键步骤：为了方便二元和三元组合，先将所有一元特征拉平/平铺
    flat_num = []
    flat_str = []
    for num_block, str_block in zip(out_num, out_str):
        # 遍历每个特征块里的所有派生项
        for i in range(num_block.shape[1]):
            flat_num.append(num_block[:, i:i+1])
            flat_str.append(str_block[i])

    # 用于存放所有最终特征的列表，首先放入拉平后的一元特征
    final_num_list = list(flat_num)
    final_str_list = list(flat_str)

    ## 2. 二元组合运算 (基于拉平后的所有一元项)
    for idx1, idx2 in itertools.combinations(range(len(flat_num)), 2):
        num_c2, str_c2 = combine(flat_num[idx1], flat_num[idx2], 
                                 [flat_str[idx1]], [flat_str[idx2]])
        # combine返回的是多列组合，需要拆开存入列表
        for i in range(num_c2.shape[1]):
            final_num_list.append(num_c2[:, i:i+1])
            final_str_list.append(str_c2[i])

    ## 3. 7.6.3 新增：三元组合运算 ((A * B) / C)
    # 由于全量组合极大，我们仅对一元特征的基础派生项做三元复合乘除，以精准捕捉目标物理方程
    for idx1, idx2, idx3 in itertools.combinations(range(len(flat_num)), 3):
        num_c3, str_c3 = combine_3(flat_num[idx1], flat_num[idx2], flat_num[idx3],
                                   flat_str[idx1], flat_str[idx2], flat_str[idx3])
        final_num_list.append(num_c3)
        final_str_list.append(str_c3[0])

    ## 整合数据
    final_num = np.hstack(final_num_list)
    final_str = np.array(final_str_list)
    
    # 定量记录空间膨胀：打印数据清洗前的特征总列数
    print(f"\n[7.6.3 空间膨胀统计] 数据清洗前，特征空间总列数为: {final_num.shape[1]}")
    
    out_data = pd.DataFrame(final_num, columns=final_str)

    ## 删除异常数据（过滤带有 NaN 和 Inf 的无效公式）
    out_data = out_data.loc[:, ~(out_data.isna().any() | \
                                  out_data.apply(np.isinf).any())]
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]

    print(f"[7.6.3 空间膨胀统计] 过滤异常与重复项后，有效特征总列数为: {out_data.shape[1]}\n")

    return out_data

def power(in_num: ND, in_name: ND, p):
    '''
    对数据进行幂运算
    '''
    if len(in_num.shape) == 1:
        in_num = in_num.reshape(-1, 1)

    out_num = np.power(in_num, p)
    out_name = [f"({n}**{p})" for n in in_name]

    return out_num, out_name

def inv_and_exp_log(in_num: ND, in_name: ND):
    '''
    7.6.3 新增：对数据进行倒数、指数、对数一元运算
    '''
    if len(in_num.shape) == 1:
        in_num = in_num.reshape(-1, 1)
        
    # 倒数 1/x (引入1e-8防止除以0)
    num_inv = 1.0 / (in_num + 1e-8)
    str_inv = [f"(1/{n})" for n in in_name]
    
    # 指数 exp(x)
    num_exp = np.exp(in_num)
    str_exp = [f"exp({n})" for n in in_name]
    
    # 对数 log|x| (加绝对值和1e-8防止负数和0报错)
    num_log = np.log(np.abs(in_num) + 1e-8)
    str_log = [f"log({n})" for n in in_name]
    
    return np.hstack((num_inv, num_exp, num_log)), np.hstack((str_inv, str_exp, str_log))

def combine(in_num1: ND, in_num2: ND, in_name1: ND, in_name2: ND):
    '''
    对数据进行二元组合运算
    '''
    if len(in_num1.shape) == 1: in_num1 = in_num1.reshape(-1, 1)
    if len(in_num2.shape) == 1: in_num2 = in_num2.reshape(-1, 1)

    num_add = in_num1 + in_num2
    str_add = [f"({n1}+{n2})" for n1, n2 in zip(in_name1, in_name2)]

    num_sub = in_num1 - in_num2
    str_sub = [f"({n1}-{n2})" for n1, n2 in zip(in_name1, in_name2)]

    num_mul = in_num1 * in_num2
    str_mul = [f"({n1}*{n2})" for n1, n2 in zip(in_name1, in_name2)]

    num_div = in_num1 / (in_num2 + 1e-8)
    str_div = [f"({n1}/{n2})" for n1, n2 in zip(in_name1, in_name2)]

    return np.hstack((num_add, num_sub, num_mul, num_div)), np.hstack((str_add, str_sub, str_mul, str_div))

def combine_3(in_num1: ND, in_num2: ND, in_num3: ND, in_name1: str, in_name2: str, in_name3: str):
    '''
    7.6.3 新增：对三个数据进行三元乘除组合运算: (A * B) / C
    '''
    # 确保列向量维度
    num_c = (in_num1 * in_num2) / (in_num3 + 1e-8)
    str_c = [f"(({in_name1}*{in_name2})/{in_name3})"]

    return num_c, str_c