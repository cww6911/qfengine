import pandas as pd
import numpy as np

def make_qfq(group):
    """
    :param group: 为未复权的历史行情数据和分红事件数据合并后的股票分组数据
    :return: 返回向后复权的历史行情价格数据
    说明：如果有以前已经用本方法计算过的复权数据，并且和当前的未复权数据的交易日期是连续的，
    则计算时取base_price为前数据最后一天的价格，
    cumprod_from = (1 + df_后复权[ret_col].iloc[1:]).cumprod()修改为-->
    cumprod_from = (1 + df_后复权[ret_col]).cumprod()
    df_后复权.loc[1:, f"{col}_qfq"] = base_price * cumprod_from 修改为-->
    df_后复权.loc[f"{col}_qfq"] = base_price * cumprod_from
    """
    df_未复权 = group.sort_values(by='date').copy()
    df_未复权 = df_未复权.reset_index(drop=True)

    df_未复权['股权系数'] = df_未复权['股权系数'].fillna(1)
    df_未复权['派息系数'] = df_未复权['派息系数'].fillna(0)

    df_未复权['last_close'] = df_未复权['close'].shift(1)
    df_未复权['last_open'] = df_未复权['open'].shift(1)
    df_未复权['last_high'] = df_未复权['high'].shift(1)
    df_未复权['last_low'] = df_未复权['low'].shift(1)

    df_未复权['per_close'] = (df_未复权['last_close']-df_未复权['派息系数'])/df_未复权['股权系数']
    df_未复权['c_change'] = (df_未复权['close']-df_未复权['per_close'])/df_未复权['per_close']

    df_未复权['per_open'] = (df_未复权['last_open']-df_未复权['派息系数'])/df_未复权['股权系数']
    df_未复权['o_change'] = (df_未复权['open']-df_未复权['per_open'])/df_未复权['per_open']

    df_未复权['per_high'] = (df_未复权['last_high']-df_未复权['派息系数'])/df_未复权['股权系数']
    df_未复权['h_change'] = (df_未复权['high']-df_未复权['per_high'])/df_未复权['per_high']

    df_未复权['per_low'] = (df_未复权['last_low']-df_未复权['派息系数'])/df_未复权['股权系数']
    df_未复权['l_change'] = (df_未复权['low']-df_未复权['per_low'])/df_未复权['per_low']

    # 初始化
    df_后复权 = df_未复权.copy().reset_index(drop=True)
    df_后复权['close_qfq'] = np.nan
    df_后复权['open_qfq'] = np.nan
    df_后复权['high_qfq'] = np.nan
    df_后复权['low_qfq'] = np.nan

    price_cols = ['close','open','high','low']
    for col in price_cols:
        # 用字符拼接方式把前面已经计算好市场真实变化率赋值给ret_col
        ret_col = f"{col[0]}_change"  # 'close' -> 'c_change'
        # 取第一天的价格作为连乘的基准价格
        # 如果是拼接以前的后复权数据，则用以前复权数据的最后一天价格作为计算起点
        base_price = df_后复权.loc[0, col]
        # 定义第一天的价格为复权价
        df_后复权.loc[0, f"{col}_qfq"] = base_price
        # 从第二天开始计算连乘因子，如果是拼接以前的后复权数据，则从第一天开始计算
        cumprod_from = (1 + df_后复权[ret_col].iloc[1:]).cumprod()
        # 从第二天开始计算复权价格，如果是拼接以前的后复权数据，则从第一天开始计算
        df_后复权.loc[1:, f"{col}_qfq"] = base_price * cumprod_from
        # 保留二位小数
        df_后复权[f"{col}_qfq"] = df_后复权[f"{col}_qfq"].round(2)

    df_后复权['change_rate'] = round(df_后复权['c_change']*100,2)
    # 选择要保留的列
    df_后复权 = df_后复权[['date','symbol','close_qfq','open_qfq','high_qfq','low_qfq'
        ,'change_rate','成交量','换手率','流通股本']]

    return df_后复权


def make_hfq(group):
    """
    :param group: 为未复权的历史行情数据和分红事件数据合并后的股票分组数据
    :return: 返回向前复权的历史行情价格数据
    特别说明：股票如果在最新交易日发生了除权，就要重新计算前复权价格，原来的计算结果是不能复用的
    """
    df_未复权 = group.sort_values(by='date').reset_index(drop=True).copy()
    # 给非除权日的'股权系数'和'派息系数'重新赋值
    df_未复权['股权系数'] = df_未复权['股权系数'].fillna(1)
    df_未复权['派息系数'] = df_未复权['派息系数'].fillna(0)
    # 生成前一个交易日的价格
    df_未复权['last_close'] = df_未复权['close'].shift(1)
    df_未复权['last_open'] = df_未复权['open'].shift(1)
    df_未复权['last_high'] = df_未复权['high'].shift(1)
    df_未复权['last_low'] = df_未复权['low'].shift(1)
    # 计算市场真实变化率
    df_未复权['per_close'] = (df_未复权['last_close']-df_未复权['派息系数'])/df_未复权['股权系数']
    df_未复权['c_change'] = (df_未复权['close']-df_未复权['per_close'])/df_未复权['per_close']

    df_未复权['per_open'] = (df_未复权['last_open']-df_未复权['派息系数'])/df_未复权['股权系数']
    df_未复权['o_change'] = (df_未复权['open']-df_未复权['per_open'])/df_未复权['per_open']

    df_未复权['per_high'] = (df_未复权['last_high']-df_未复权['派息系数'])/df_未复权['股权系数']
    df_未复权['h_change'] = (df_未复权['high']-df_未复权['per_high'])/df_未复权['per_high']

    df_未复权['per_low'] = (df_未复权['last_low']-df_未复权['派息系数'])/df_未复权['股权系数']
    df_未复权['l_change'] = (df_未复权['low']-df_未复权['per_low'])/df_未复权['per_low']

    # 初始化
    df_未复权['close_hfq'] = np.nan
    df_未复权['open_hfq'] = np.nan
    df_未复权['high_hfq'] = np.nan
    df_未复权['low_hfq'] = np.nan
    # 对原数据进行降序排序
    df_前复权 = df_未复权.sort_values(by='date', ascending=False).reset_index(drop=True).copy()
    # 释放缓存
    del df_未复权
    # 向量化：计算从最新日到当前日的“未来累积因子”的倒数
    price_cols = ['close', 'open', 'high', 'low']
    for col in price_cols:
        base_price = df_前复权.loc[0, col]
        rev_ret = f"{col[0]}_change"  # 'close' -> 'c_change'
        future_cumprod = np.concatenate([[1.0], (1 + df_前复权[rev_ret].iloc[0:len(df_前复权)-1]).cumprod()])
        forward_factor = 1.0 / future_cumprod
        df_前复权[f"{col}_hfq"] = base_price * forward_factor
        # 保留二位小数
        df_前复权[f"{col}_hfq"] = df_前复权[f"{col}_hfq"].round(2)

    df_前复权['change_rate'] = round(df_前复权['c_change'] * 100, 2)

    df_前复权 = df_前复权[['date','symbol','close_hfq','open_hfq','high_hfq','low_hfq'
        ,'change_rate','成交量','换手率','流通股本']]

    df_前复权 = df_前复权.sort_values(by='date').reset_index(drop=True)

    return df_前复权