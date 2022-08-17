"""
番外篇：详解画图函数 | 邢不行 | 2021股票量化课程
author: 邢不行
微信: xbx719
"""
import pandas as pd
from program.Function import *

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 定义使用的字体，是个数组。
plt.rcParams['axes.unicode_minus'] = False

# ===== 读入基础数据
df = pd.read_csv(root_path + '/data/small_market_value_week_3.csv', encoding='gbk', parse_dates=['交易日期'])

# ===== figure:创建或者激活一个画布
# 设置画布大小
fig1 = plt.figure(figsize=(5, 5))
print('这是图片1', fig1)
fig2 = plt.figure(figsize=(5, 15))
print('这是图片2', fig2)
fig3 = plt.figure(figsize=(15, 5))
print('这是图片3', fig3)
print('当前画布：', plt.get_fignums())
# plt.figure(fig2)

# ===== plot:绘制图片x,y,label
plt.plot(df['交易日期'], df['equity_curve'], label='策略净值', linewidth=1)
plt.plot(df['交易日期'], df['benchmark'], label='基准净值', linewidth=3)

# ===== legend:设置图例
# plt.legend(loc=0)
# plt.legend(loc=0, fontsize=25)
# plt.tick_params(labelsize=25)

# ===== show:展示图片
# fig1.show()
# print('fig show完之后的画布：', plt.get_fignums())
# exit()
# plt.show()
# print('plt show完之后的画布：', plt.get_fignums())
# exit()

# ===== twinx：创建右轴对象
ax_r = plt.twinx()
ax_r.plot(df['交易日期'], df['涨跌幅'], 'y', label='涨跌幅-右轴', linewidth=1)
ax_r.legend(loc=1, fontsize=25)
ax_r.tick_params(labelsize=25)
plt.show()
