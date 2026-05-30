from expansion import expand
from result_sorting import sort_result
from result_displaying import display_result
from sis_ana import sis
from coefficients_fitting import fit
from result_plotting import plot_result
import pandas as pd
import numpy as np
import torch
import os

def stop_or_not(r2_history, epoch, max_epoch=10, tol=1e-4):
    '''
    模块 3：迭代终止策略函数
    '''
    if epoch > max_epoch:
        print(">>> 触发最大演化轮数，停止迭代。")
        return True
    if len(r2_history) > 0 and r2_history[-1] > 0.995:
        print(">>> 目标方程已捕获 (R^2 > 0.995)，动态提前终止！")
        return True
    if len(r2_history) >= 2:
        if abs(r2_history[-1] - r2_history[-2]) < tol:
            print(">>> 连续两轮最高分无显著提升，算法已饱和停机。")
            return True
    return False

def expand_next(ori_df, top_features_results, data_sis_df):
    '''
    模块 1：引导函数。
    安全、标准地提取纯复合特征（不带拟合系数），彻底解决 SymPy 报错与 KeyError
    '''
    extracted_pure_dfs = []
    
    # 提取排名前 5 的精英特征模型
    for res in top_features_results[:5]:
        # 通过类私有属性的后门或者公开方法获取纯特征 DataFrame
        # 自定义类中名为 self.__data 的属性储存了它原本的纯 DataFrame (含纯列名和纯数值)
        # 在 Python 中，可以通过 _类名__属性名 强制访问私有成员
        pure_feature_df = res._FeatureInformation__data
        extracted_pure_dfs.append(pure_feature_df)
        
    # 将这 5 个纯特征 DataFrame 横向拼接到一块
    top_features_df = pd.concat(extracted_pure_dfs, axis=1)
    
    # 动态拼接：最原始的 5 个物理量种子项 + 这一轮筛选出的 5 个纯高级复合特征
    next_pool = pd.concat([ori_df, top_features_df], axis=1)
    
    # 去除由于多轮循环产生的同名重复特征列
    next_pool = next_pool.loc[:, ~next_pool.columns.duplicated()]
    
    return next_pool

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.getcwd()

    # 1. 读取种子数据
    ori_data = pd.read_csv(os.path.join(path, "data.csv"))
    focus = pd.read_csv(os.path.join(path, "focus.csv"))
    y_true = focus.to_numpy()

    # 初始化动态演化特征池
    X_pool = ori_data.copy()
    
    r2_history = []
    epoch = 1
    max_display = 10 # 每一轮展示前10名

    print("================== 开始智能符号演化流程 ==================")

    while not stop_or_not(r2_history, epoch):
        print(f"\n>>>>>> 正在进行第 [ {epoch} ] 轮公式演化，当前特征池项数: {X_pool.shape[1]}")
        
        # 扩展特征空间
        data_expanded = expand(X_pool)
        
        # 确定性独立筛选 (SIS) 过滤出高潜力的 30 个复合特征，防止后续计算过慢
        data_sis = sis(data_expanded, y_true, 30)
        
        # 系数拟合 
        r2, coef, loss = fit(data_sis, y_true, device)
        
        # 结果排序 (sort_result 内部会实例化 FeatureInformation 类并返回其对象列表)
        results = sort_result(data_sis.to_numpy(), 
                              data_sis.columns.to_numpy(),
                              r2, coef, loss, max_display)
        
        # 核心适配：使用 .get_r2() 获取特征信息对象的 R2 值
        current_best_r2 = results[0].get_r2()
        r2_history.append(current_best_r2)
        print(f"第 [ {epoch} ] 轮演化完成！当前最佳公式 R^2 = {current_best_r2:.4f}")
        
        # 核心适配：使用 .get_full_name() 直接获取带拟合系数的完整公式形态 (如 a*model+b)
        print(f"当前最强公式形态: {results[0].get_full_name()}")

        # 如果已经是最后一轮或满足终止条件，输出最终日志并绘图
        if stop_or_not(r2_history, epoch + 1) or current_best_r2 > 0.995:
            print("\n================== 演化收敛，生成最终报告 ==================")
            display_result(results, max_display, path)
            plot_result(results[0], focus, path)
            break
            
        # 提取本轮精锐，回代拼接，更新特征池以进入下一轮迭代
        X_pool = expand_next(ori_data, results, data_sis)
        epoch += 1

if __name__ == "__main__":
    main()