"""
1. 因子处理：缺失值处理、去极值、标准化、PCA降维、中性化（用线性回归剔除因子间相关度高的部分）

2. 因子有效性分析：因子IC分析（确定因子和收益率之间的相关性）
IC（信息系数）：某一期的IC指的是该期因子暴露值和股票下期的实际回报值在横截面上的相关系数
因子暴露度-处理（缺失值处理、去极值。标准化）后的因子值，股票下期的实际回报值-下期收益率，相关系数-斯皮尔曼相关系数

3. 因子收益率k：因子收益率 * 因子暴露度 + b = 下期收益率

4. 多因子相关性分析：还是使用斯皮尔曼秩相关系数，但是对象是两个因子的IC值序列分析

5. 多因子选股最常用的方法就是打分法和回归法

6. 收益指标：回测收益，回测年化收益，基准收益，基准年化收益
   风险指标：最大回撤越小越好（30%以内）， 夏普比率越大越好（1以上）
"""
import pandas as pd
import numpy as np
import scipy.stats as st
from alphalens import tears, performance, plotting, utils

df = pd.DataFrame([[1, 2], [4, 5]], columns=["A", "B"])

# 计算斯皮尔相关系数Rank IC，取值 [-1, 1]之间
print(st.spearmanr(df["A"], df["B"]))

"""使用alphalens更简易的做因子分析"""
# 输入因子表和收盘价表到返回到期收益率表，再将因子表和到期收益表整合返回综合因子数据表
factor_data = utils.get_clean_factor_and_forward_returns("factor", "price")
# 因子IC的计算
IC = performance.factor_information_coefficient(factor_data)
# 因子时间序列和移动平均图，看出一个因子在时间上的正负性、
plotting.plot_ic_ts(IC)
# 因子分布直方图，IC平均值，标准差
plotting.plot_ic_hist(IC)
# 热力图
mean_monthly_ic = performance.mean_information_coefficient(factor_data, by_time="1m")
plotting.plot_monthly_ic_heatmap(mean_monthly_ic)
# IC分析合集
tears.create_information_tear_sheet(factor_data)

# 收益率分析
tears.create_returns_tear_sheet(factor_data)
# 因子的每一期的收益（因子收益）
performance.factor_returns(factor_data).iloc[:, 0].mean()
