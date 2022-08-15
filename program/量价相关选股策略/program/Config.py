"""
量价相关策略 | 邢不行 | 2021股票量化课程
微信号：xbx9585
"""
import os

# ===选股参数设定
period = 'W'  # W代表周，M代表月

# ===获取项目根目录
_ = os.path.abspath(os.path.dirname(__file__))  # 返回当前文件路径
root_path = os.path.abspath(os.path.join(_, '..'))  # 返回根目录文件夹

stock_data_path = '/Users/lishechuan/python/coincock/data/stock/'
back_test_start = '20170123'
back_test_end = '20211013'
