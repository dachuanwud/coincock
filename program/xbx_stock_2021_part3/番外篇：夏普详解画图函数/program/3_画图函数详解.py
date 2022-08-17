"""
番外篇：详解画图函数 | 邢不行 | 2021股票量化课程
author: 邢不行
微信: xbx719
"""
import pandas as pd
from program.Function import *

# 读入基础数据
df = pd.read_csv(root_path + '/data/small_market_value_week_3.csv', encoding='gbk', parse_dates=['交易日期'])
data_dict = {
    '策略净值': 'equity_curve',
    '基准净值': 'benchmark',
}
# draw_equity_curve_mat(df, data_dict, date_col='交易日期', right_axis={'指数涨跌幅': '指数涨跌幅'}, log=True)
draw_equity_curve_plotly(df, data_dict, date_col='交易日期', right_axis={'指数涨跌幅': '指数涨跌幅'}, log=True)
