import pandas as pd
from Functions import *
from Config import *
from multiprocessing import Pool, freeze_support, cpu_count
from datetime import datetime
import platform
import glob

def boll_buy_sell_Strategy(df):
    condition1 = df['收盘价_复权'] < df['lower']
    condition2 = df['收盘价_复权'].shift(1) <= df['lower'].shift(1)
    df.loc[condition1 & condition2, 'signal'] = 1
    df['signal'].fillna(value=0, inplace=True)
    return df