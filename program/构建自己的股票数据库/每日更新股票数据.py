"""
《邢不行-2019新版|Python股票量化投资课程》
author：邢不行
微信：xingbuxing0807

更新于：2021-02-26
主要更正了由于数据接口导致的两个问题

更新于：2022-01-06
主要更新了获取数据的接口，新增市值列

本节课讲解如何结合历史数据，每天获取股票的数据，
构建完整实时股票数据库。

"""
import os
import operator
import requests
import time
from datetime import datetime
from requests.adapters import HTTPAdapter
import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
# 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)


def requestForNew(url, max_try_num=10, sleep_time=5):
    headers = {
        'Referer': 'http://finance.sina.com.cn',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.62'
    }
    for i in range(max_try_num):
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response
        else:
            print("链接失败", response)
            time.sleep(sleep_time)


def getDate():
    url = 'https://hq.sinajs.cn/list=sh000001'
    response = requestForNew(url).text
    data_date = str(response.split(',')[-4])
    # 获取上证的指数日期
    return data_date


# 通过新浪财经获取每日更新的股票代码
def getStockCodeForEveryday():
    df = pd.DataFrame()
    page = 1
    while True:
        # 1~100页，不用担心每天新增
        url = 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=' \
              + str(page) + '&num=80&sort=changepercent&asc=0&node=hs_a&symbol=&_s_r_a=page'
        # print(url)
        content = requestForNew(url).json()
        if not content:
            # if content =[]: 这个写法也可以
            print("股票信息，获取完毕。")
            break
        print("正在读取页面" + str(page))
        time.sleep(3)
        df = df.append(pd.DataFrame(content, dtype='float'), ignore_index=True)
        page += 1

    rename_dict = {'symbol': '股票代码', 'code': '交易日期', 'name': '股票名称', 'open': '开盘价',
                   'settlement': '前收盘价', 'trade': '收盘价', 'high': '最高价', 'low': '最低价',
                   'buy': '买一', 'sell': '卖一', 'volume': '成交量', 'amount': '成交额',
                   'changepercent': '涨跌幅', 'pricechange': '涨跌额',
                   'mktcap': '总市值', 'nmc': '流通市值', 'ticktime': '数据更新时间', 'per': 'per', 'pb': '市净率',
                   'turnoverratio': '换手率'}
    df.rename(columns=rename_dict, inplace=True)
    tradeDate = getDate()
    df['交易日期'] = tradeDate
    df['总市值'] = df['总市值'] * 10000
    df['流通市值'] = df['流通市值'] * 10000
    df = df[['股票代码', '股票名称', '交易日期', '开盘价', '最高价', '最低价', '收盘价', '前收盘价', '成交量', '成交额', '流通市值', '总市值']]
    # 把转化成float的code替换成交易日期
    return df


def get_project():
    print(os.getcwd())                    # 获取当前工作目录路径
    print (os.path.abspath('.'))           # 获取当前工作目录路径
    print (os.path.abspath('test.txt'))    # 获取当前目录文件下的工作目录路径
    print (os.path.abspath('..'))          # 获取当前工作的父目录 ！注意是父目录路径
    print (os.path.abspath(os.curdir))     # 获取当前工作目录路径

get_project()
df = getStockCodeForEveryday()
df = df[df['开盘价'] - 0 > 0.00001]
df.reset_index(drop=True, inplace=True)

print(df)

for i in df.index:
    t = df.iloc[i:i + 1, :]
    stock_code = t.iloc[0]['股票代码']
    print(stock_code)
    continue
    # 构建存储文件路径
    path = '/Users/lishechuan/python/coincock/data/stock/' \
           + stock_code + '.csv'
    # 文件存在，不是新股
    if os.path.exists(path):
        print(path)
        t.to_csv(path, header=None, index=False, mode='a', encoding='gbk')
    # 文件不存在，说明是新股
    else:
        # 先将头文件输出
        print(path)
        pd.DataFrame(columns=['数据由李涉川整理']).to_csv(path, index=False, encoding='gbk')
        t.to_csv(path, index=False, mode='a', encoding='gbk')
    print(stock_code)
