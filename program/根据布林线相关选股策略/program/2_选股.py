"""
量价相关策略 | 邢不行 | 2021股票量化课程
微信号：xbx9585
"""
from Config import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Functions import *
from Signal import *

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
# 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False

# ===手续费
c_rate = 1.2 / 10000  # 手续费
t_rate = 1 / 1000  # 印花税
select_stock_num = 3  # 选股数，当选股数为None时，会等权买入所有，当选股数小于0时，会看做百分比，用百分比进行选股。

# ===导入数据
# 导入指数数据
index_data = import_index_data('/Users/lishechuan/python/coincock/data/指数数据/sh000300.csv', start=back_test_start, end=back_test_end)

# 创造空的事件周期表，用于填充不选股的周期
empty_df = create_empty_data(index_data, period)

# 从pickle文件中读取整理好的所有股票数据
df = pd.read_pickle(root_path + '/data/数据整理/all_stock_data_' + period + '.pkl')
df.dropna(subset=['下周期每天涨跌幅'], inplace=True)
# ===删除下个交易日不交易、开盘涨停的股票，因为这些股票在下个交易日开盘时不能买入。
df = df[df['下日_是否交易'] == 1]
df = df[df['下日_开盘涨停'] == False]
df = df[df['下日_是否ST'] == False]
df = df[df['下日_是否退市'] == False]

# ****************************以下内容可以改动****************************
# 选股的方法请写在这里
# factor = '量价相关性'  # 量价相关性
# ascending = True  # True，从小到大    False，从大到小
# df.dropna(subset=[factor], inplace=False)
# df['排名'] = df.groupby('交易日期')[factor].rank(ascending=ascending, method='first')
# df['排名_百分比'] = df.groupby('交易日期')[factor].rank(ascending=ascending, pct=True, method='first')
df['排名'] = df.groupby('交易日期')['总市值'].rank()
df = df[df['排名'] <= select_stock_num]
df = boll_buy_sell_Strategy(df)
df.reset_index(inplace=True, drop=True)
# print(df.tail(10))
# exit()
# ****************************以上内容可以改动****************************

# ===选股
# if select_stock_num:
#     if select_stock_num >= 1:
#         df = df[df['排名'] <= select_stock_num]
#     else:
#         df = df[df['排名_百分比'] <= select_stock_num]

# ===按照开盘买入的方式，修正选中股票在下周期每天的涨跌幅。
# 即将下周期每天的涨跌幅中第一天的涨跌幅，改成由开盘买入的涨跌幅
# 按照开盘买入的方式，修正选中股票在下周期每天的涨跌幅。
# 即将下周期每天的涨跌幅中第一天的涨跌幅，改成由开盘买入的涨跌幅
df['下日_开盘买入涨跌幅'] = df['下日_开盘买入涨跌幅'].apply(lambda x: [x])
df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(lambda x: x[1:])
df['下周期每天涨跌幅'] = df['下日_开盘买入涨跌幅'] + df['下周期每天涨跌幅']
print(df[['交易日期', '股票名称', '下日_开盘买入涨跌幅', '下周期每天涨跌幅']].head(6))

# ===整理选中股票数据
# 挑选出选中股票
df['股票代码'] += ' '
df['股票名称'] += ' '
group = df.groupby('交易日期')
select_stock = pd.DataFrame()
select_stock['买入股票代码'] = group['股票代码'].sum()
select_stock['买入股票名称'] = group['股票名称'].sum()

# 计算下周期每天的资金曲线
select_stock['选股下周期每天资金曲线'] = group['下周期每天涨跌幅'].apply(lambda x: np.cumprod(np.array(list(x)) + 1, axis=1).mean(axis=0))

# 扣除买入手续费
select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'] * (1 - c_rate)  # 计算有不精准的地方
# 扣除卖出手续费、印花税。最后一天的资金曲线值，扣除印花税、手续费
select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'].apply(
    lambda x: list(x[:-1]) + [x[-1] * (1 - c_rate - t_rate)])

# 计算下周期整体涨跌幅
select_stock['选股下周期涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(lambda x: x[-1] - 1)
# 计算下周期每天的涨跌幅
select_stock['选股下周期每天涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(
    lambda x: list(pd.DataFrame([1] + x).pct_change()[0].iloc[1:]))
del select_stock['选股下周期每天资金曲线']

# 计算整体资金曲线
select_stock.reset_index(inplace=True)
select_stock['资金曲线'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()
print(select_stock.tail(50))
select_stock.set_index('交易日期', inplace=True)
empty_df.update(select_stock)
select_stock = empty_df
select_stock.reset_index(inplace=True, drop=False)

# ===计算选中股票每天的资金曲线
# 计算每日资金曲线
equity = pd.merge(left=index_data, right=select_stock[['交易日期', '买入股票代码']], on=['交易日期'],
                  how='left', sort=True)  # 将选股结果和大盘指数合并

equity['持有股票代码'] = equity['买入股票代码'].shift()
equity['持有股票代码'].fillna(method='ffill', inplace=True)
equity.dropna(subset=['持有股票代码'], inplace=True)
del equity['买入股票代码']
equity['涨跌幅'] = select_stock['选股下周期每天涨跌幅'].sum()
equity['equity_curve'] = (equity['涨跌幅'] + 1).cumprod()
equity['benchmark'] = (equity['指数涨跌幅'] + 1).cumprod()
print(equity.tail())

# 计算每周选股超额收益
equity.loc[equity['持有股票代码'] != equity['持有股票代码'].shift(), '周期开始时间'] = equity['交易日期']
equity['周期开始时间'].fillna(method='ffill', inplace=True)
period_df = pd.DataFrame()
period_df['选中股票代码'] = equity.groupby('周期开始时间')['持有股票代码'].first()
period_df['选股涨跌幅'] = equity.groupby('周期开始时间')['涨跌幅'].apply(lambda x: (1 + x).prod() - 1)
period_df['指数涨跌幅'] = equity.groupby('周期开始时间')['指数涨跌幅'].apply(lambda x: (1 + x).prod() - 1)
period_df['超额收益'] = period_df['选股涨跌幅'] - period_df['指数涨跌幅']

# ===计算策略评价指标
rtn, year_return, month_return = strategy_evaluate(equity, select_stock)
print(rtn)
print(year_return)

# 绘制策略曲线
equity = equity.reset_index()
# ===画图
draw_equity_curve_mat(equity, data_dict={'策略涨跌幅': 'equity_curve', '基准涨跌幅': 'benchmark'}, date_col='交易日期')
# 如果上面的函数不能画图，就用下面的画图
# draw_equity_curve_plotly(equity, data_dict={'策略涨跌幅': 'equity_curve', '基准涨跌幅': 'benchmark'}, date_col='交易日期')
