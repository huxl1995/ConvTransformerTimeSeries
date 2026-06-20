import os

import akshare as ak

# 获取贵州茅台（600519）的历史行情数据
# # adjust='hfq' 表示后复权，'' 则为不复权
# stock_hfq_df = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20250201", end_date="20260601", adjust="hfq")
# print(stock_hfq_df.head())



import baostock as bs
import pandas as pd

# 登陆系统
lg = bs.login()

# 获取沪深A股历史K线数据
# frequency="d"（日线），adjustflag="3"（后复权）
rs = bs.query_history_k_data_plus("sh.600519",
    "date,code,open,high,low,close,volume,amount,turn",
    start_date="2025-01-01", end_date="2026-06-01",
    frequency="d", adjustflag="3")

data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
print(result.head())

# 登出系统
bs.logout()