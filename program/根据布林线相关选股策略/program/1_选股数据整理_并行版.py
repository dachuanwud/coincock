"""
量价相关策略 | 邢不行 | 2021股票量化课程
微信号：xbx9585
"""
import pandas as pd
from Functions import *
from Config import *
from multiprocessing import Pool, freeze_support, cpu_count
from datetime import datetime
import platform
import glob

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数

# ===读取准备数据
# 读取所有股票代码的列表
stock_code_list = get_stock_code_list_in_one_dir(stock_data_path)
print('股票数量：', len(stock_code_list))

# 导入指数数据
index_data = import_index_data('/Users/lishechuan/python/coincock/data/指数数据/sh000300.csv', start=back_test_start, end=back_test_end)


# ===循环读取并且合并
def calculate_by_stock(code):
    """
    整理数据核心函数
    :param code: 股票代码
    :return: 一个包含该股票所有历史数据的DataFrame
    """
    print(code)

    # =读入A股数据
    path = stock_data_path + '%s.csv' % code
    df = pd.read_csv(path, encoding='gbk', skiprows=1, parse_dates=['交易日期'])

    # 计算换手率
    df['换手率'] = df['成交额'] / df['流通市值']
    # =计算涨跌幅
    df['涨跌幅'] = df['收盘价'] / df['前收盘价'] - 1
    # 计算复权因子：假设你一开始有1元钱，投资到这个股票，最终会变成多少钱。
    df['复权因子'] = (1 + df['涨跌幅']).cumprod()
    # 计算后复权价
    df['收盘价_复权'] = df['复权因子'] * (df.iloc[0]['收盘价'] / df.iloc[0]['复权因子'])
    df['开盘买入涨跌幅'] = df['收盘价'] / df['开盘价'] - 1  # 为之后开盘买入做好准备

    # =计算交易天数
    df['上市至今交易天数'] = df.index + 1

    # 需要额外保存的字段
    extra_fill_0_list = []  # 在和上证指数合并时使用。
    extra_agg_dict = {}  # 在转换周期时使用。

    # =将股票和上证指数合并，补全停牌的日期，新增数据"是否交易"、"指数涨跌幅"
    df = merge_with_index_data(df, index_data, extra_fill_0_list)
    if df.empty:
        return pd.DataFrame()
    # =计算涨跌停价格
    df = cal_zdt_price(df)

    # ==== 计算因子
    df['量价相关性'] = df['收盘价_复权'].rolling(10).corr(df['换手率'])
    extra_agg_dict['量价相关性'] = 'last'
    extra_agg_dict['总市值'] = 'last'
    extra_agg_dict['流通市值'] = 'last'

    # ==== 计算因子

    # =计算下个交易的相关情况
    df['下日_是否交易'] = df['是否交易'].shift(-1)
    df['下日_一字涨停'] = df['一字涨停'].shift(-1)
    df['下日_开盘涨停'] = df['开盘涨停'].shift(-1)
    df['下日_是否ST'] = df['股票名称'].str.contains('ST').shift(-1)
    df['下日_是否S'] = df['股票名称'].str.contains('S').shift(-1)
    df['下日_是否退市'] = df['股票名称'].str.contains('退').shift(-1)
    df['下日_开盘买入涨跌幅'] = df['开盘买入涨跌幅'].shift(-1)

    # =将日线数据转化为月线或者周线
    df = transfer_to_period_data(df, period, extra_agg_dict)

    # =对数据进行整理
    # 删除上市的第一个周期
    df.drop([0], axis=0, inplace=True)  # 删除第一行数据
    # 计算下周期每天涨幅
    df['下周期每天涨跌幅'] = df['每天涨跌幅'].shift(-1)
    df['下周期涨跌幅'] = df['涨跌幅'].shift(-1)
    del df['每天涨跌幅']

    # =删除不能交易的周期数
    # 删除月末为st状态的周期数
    df = df[df['股票名称'].str.contains('ST') == False]
    # 删除月末为s状态的周期数
    df = df[df['股票名称'].str.contains('S') == False]
    # 删除月末有退市风险的周期数
    df = df[df['股票名称'].str.contains('退') == False]
    # 删除月末不交易的周期数
    df = df[df['是否交易'] == 1]
    # 删除交易天数过少的周期数
    df = df[df['交易天数'] / df['市场交易天数'] >= 0.8]
    df.drop(['交易天数', '市场交易天数'], axis=1, inplace=True)

    return df  # 返回计算好的数据


# ===并行提速的办法
if __name__ == '__main__':

    # 测试
    #calculate_by_stock('sz300479')

    # 标记开始时间
    start_time = datetime.now()

    # 并行处理
    multiply_process = True
    if multiply_process:
        with Pool(max(cpu_count() - 2, 1)) as pool:
            df_list = pool.map(calculate_by_stock, sorted(stock_code_list))
    # 传行处理
    else:
        df_list = []
        for stock_code in stock_code_list:
            data = calculate_by_stock(stock_code)
            df_list.append(data)
    print('读入完成, 开始合并，消耗时间', datetime.now() - start_time)

    # 合并为一个大的DataFrame
    all_stock_data = pd.concat(df_list, ignore_index=True)
    all_stock_data.sort_values(['交易日期', '股票代码'], inplace=True)  # ===将数据存入数据库之前，先排序、reset_index
    all_stock_data.reset_index(inplace=True, drop=True)

    # 将数据存储到pickle文件
    all_stock_data.to_pickle('/Users/lishechuan/python/coincock/program/根据布林线相关选股策略/data/数据整理/all_stock_data_' + period + '.pkl')

