"""
《邢不行-2019新版|Python股票量化投资课程》
author：邢不行
微信：xingbuxing0807

一键卖出十个卖出所有股票的所有持仓
"""
import easytrader
import pandas as pd
import time
import warnings

pd.set_option('expand_frame_repr', False)
warnings.filterwarnings("ignore")

def get_today_data_from_sinajs(_stock_code_list):
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

    # =====构建网址
    url = "https://hq.sinajs.cn/list=" + ",".join(_stock_code_list)

    # =====抓取数据
    content = requestForNew(url).text  # 使用python自带的库，从网络上获取信息

    # =====将数据转换成DataFrame
    content = content.strip()  # 去掉文本前后的空格、回车等
    data_line = content.split('\n')  # 每行是一个股票的数据
    data_line = [i.replace('var hq_str_', '').split(',') for i in data_line]
    df = pd.DataFrame(data_line, dtype='float')  #

    # =====对DataFrame进行整理
    df[0] = df[0].str.split('="')
    df['stock_code'] = df[0].str[0].str.strip()
    df['stock_name'] = df[0].str[-1].str.strip()
    df['candle_end_time'] = df[30] + ' ' + df[31]  # 股票市场的K线，是普遍以当跟K线结束时间来命名的
    df['candle_end_time'] = pd.to_datetime(df['candle_end_time'])

    rename_dict = {6: 'buy1', 7: 'sell1'}  # 自己去对比数据，会有新的返现
    # 其中amount单位是股，volume单位是元
    df.rename(columns=rename_dict, inplace=True)
    df = df[
        ['stock_code','buy1', 'sell1']]
    return df

# =====客户端初始化
user = easytrader.use('yh_client')  # 选择银河客户端

# 输入用户名和密码，以及程序的路径
user.prepare(
    user='', password='',
    exe_path='C:\\双子星-中国银河证券\\xiadan.exe'
)

# =====获取账户资金状况
balance = pd.DataFrame(user.balance)
print('\n账户资金状况：')
print(balance)

# =====获取持仓
position_info = pd.DataFrame(user.position)
if position_info.empty:
    print('没有持仓')
    exit()
else:
    print(position_info)
time.sleep(1)

slippery_rate = 2 / 1000  # 设置下单运行的滑点

for index, row in position_info.iterrows():
    print()

    amount = row['当前持仓']
    security_code = row['证券代码']
    print('>' * 5, '准备下单卖出股票', row['证券代码'], amount, '<' * 5)

    try:
        result = user.sell_market(
            security_code, '%s' % amount
        )  # 这边可以优化买入数量
    except easytrader.exceptions.TradeError as e:
        print(security_code, '交易失败', str(e))
        continue

    print(security_code, '卖出股票成功：', result)

# =====获取今日委托
today_entrusts = pd.DataFrame(user.today_entrusts)
print('\n今日委托：')
print(today_entrusts)

# =====查看今日成交
today_trades = pd.DataFrame(user.today_trades)
print('\n今日成交：')
print(today_trades)
