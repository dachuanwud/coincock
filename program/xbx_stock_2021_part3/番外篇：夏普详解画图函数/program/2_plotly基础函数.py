"""
番外篇：详解画图函数 | 邢不行 | 2021股票量化课程
author: 邢不行
微信: xbx719
"""
import pandas as pd
from program.Function import *

# ===== 读入基础数据
df = pd.read_csv(root_path + '/data/small_market_value_week_3.csv', encoding='gbk', parse_dates=['交易日期'])

# ===== make_subplots:创建子画布
fig1 = make_subplots()
print('这是图片1', fig1)

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
print('这是图片2', fig2)

# ===== Scatter:创建散点图对象
scatter_object1 = go.Scatter(x=df['交易日期'], y=df['equity_curve'], name='策略净值')
scatter_object2 = go.Scatter(x=df['交易日期'], y=df['benchmark'], name='基准净值')

# ===== add_trace:添加轨迹
fig2.add_trace(scatter_object1)
fig2.add_trace(scatter_object2)

# ===== update_layout：更新布局
# fig2.update_layout(template="none", width=1600, height=900, title_text='这是图片的名字')
fig2.update_layout(template="none", width=1600, height=900, title_text='这是图片的名字', hovermode='x')

# ===== plot:绘制图片
save_path = root_path + '/data/pic.html'
# plot(figure_or_data=fig2, filename=save_path)
# exit()
plot(figure_or_data=fig2, filename=root_path + '/data/pic.html', auto_open=False)

# ===== 通过命令行打开图片
res = os.system('start ' + save_path)
if res != 0:
    os.system('open ' + save_path)
