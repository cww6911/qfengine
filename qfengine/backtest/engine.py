import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas_market_calendars import get_calendar
import logging
from collections import defaultdict
import traceback
from logging.handlers import RotatingFileHandler
# 创建日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # 设置最低日志级别为 DEBUG

# 创建控制台处理器并设置级别为INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# 创建带轮转的日志文件处理器
file_handler = RotatingFileHandler('app.log', maxBytes=1024*1024, backupCount=3,encoding='utf-8')  # 每个日志文件最大1MB，最多保留5个备份文件
file_handler.setLevel(logging.INFO)

# 创建格式化器并添加到处理器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理器添加到记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logging.getLogger('matplotlib').setLevel(logging.INFO)


class PositionManager:
    def __init__(self, **params):
        # 帐户起始资金
        self.total_capital = 1000000
        # 帐户当前可用资金
        self.current_capital = self.total_capital
        self.params = params
        # 持仓信息
        self.positions = {}
        self.buy_commission = 0.0005
        self.sell_commission = 0.0015
        #self.spread = 0.0005
        self.basis_index = '000300'
        self.portfolio_value = 0
        self.position_value = 0
        self.position_days = 10
        self.stock_pool = []
        self.trade_status = defaultdict(list) # 用于记录每日的成交信息，key为成交日期


class DataSource:

    def __init__(self,data):

        data = data.copy()

        # 检查必要列是否存在（英文或中文）
        required_cols = {'date', 'symbol','open','close','high','low'}
        cols = set(data.columns)
        if not required_cols.issubset(cols):
            raise ValueError(f"缺失必要列: {required_cols - cols}")

        data['date'] = pd.to_datetime(data['date'])
        data = data.drop_duplicates(subset=['symbol', 'date'], keep='last')

        # 【关键】按 symbol 分组，预存为字典（或保留原表+索引）
        self.symbol_dfs = {
            symbol: group.set_index('date', drop=False).sort_index()
            for symbol, group in data.groupby('symbol')
        }

    def _get_price(self, symbol, date, price_type):

        stock_df = self.symbol_dfs.get(symbol)
        if stock_df is None:
            logging.warning(f"股票 {symbol} 不存在于数据中")
            return None

        date_dt = pd.to_datetime(date)

        try:
            return stock_df.loc[date_dt, price_type]
        except KeyError:
            # 日期不存在
            return None

    def get_open(self, symbol, date):
        return self._get_price(symbol, date, 'open')

    def get_close(self, symbol, date):
        return self._get_price(symbol, date, 'close')

    def get_high(self, symbol, date):
        return self._get_price(symbol, date, 'high')

    def get_low(self, symbol, date):
        return self._get_price(symbol, date, 'low')


class BacktestEngine(PositionManager,DataSource):
    def __init__(self, start_date=None, end_date=None,data=None,run_func=None,auto_adjust = False):
        if data is None:
            raise ValueError("Data must be provided for DataSource initialization")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        PositionManager.__init__(self)  # 调用 PositionManager 的初始化方法
        DataSource.__init__(self,data)  # 调用 DataSource 的初始化方法
        self.trades = []  # 交易记录
        #self.trading_dates_index = 0
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop_duplicates(subset=['symbol', 'date'], keep='last')
        self.data = data
        self.auto_adjust = auto_adjust
        self.stop_loss_return = None
        self.results = None
        self.run_func = run_func if run_func else self.default_run
        try:
            if start_date is not None and end_date is not None:
                self.start_date = pd.to_datetime(start_date).tz_localize(None)
                self.end_date = pd.to_datetime(end_date).tz_localize(None)
                self.exchange = get_calendar('SSE')
                self.trading_days = self.exchange.valid_days(start_date=self.start_date,end_date=self.end_date).tz_localize(None)
                #self.all_trading_days = self.exchange.valid_days(start_date='2000-08-01',end_date=self.end_date).tz_localize(None)
                self.trading_days_set = set(self.trading_days)
                self.bool = True
                # 确保 current_date 是一个交易日
                if self.start_date in self.trading_days_set:
                    self.current_date = self.start_date
                else:
                    self.current_date = self.get_next_trading_day(self.start_date)
                self.results = [{'date': self.start_date, 'portfolio_value': self.total_capital}]
                self.benchmark_data = self.download_benchmark_data()
                self.run_backtest()
            else:
                raise ValueError("must input start_date and end_date")
        except ValueError:
            raise ValueError("Invalid date format, use YYYY-MM-DD")

    def download_benchmark_data(self):
        symbol = self.basis_index
        # 确保日期格式为 YYYYMMDD
        start_date_str = pd.to_datetime(self.start_date).strftime('%Y%m%d')
        end_date_str = pd.to_datetime(self.end_date).strftime('%Y%m%d')
        try:
            # 下载沪深300指数的历史数据
            benchmark_data = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start_date_str, end_date=end_date_str)
            benchmark_data = benchmark_data[['日期', '收盘']]
            benchmark_data = benchmark_data.rename(columns={'日期': 'date', '收盘': 'close'})
            benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
            benchmark_data.set_index('date', inplace=True)
            return benchmark_data
        except Exception as e:
            logging.error(f"Error downloading benchmark data: {e}")
            return None

    def get_benchmark_close(self):
        #获取基准指数的收盘价
        benchmark_data = self.benchmark_data
        if benchmark_data is not None and self.current_date in benchmark_data.index:
            benchmark_close = benchmark_data.loc[self.current_date, 'close']
        else:
            logging.warning(f"No benchmark data available for date: {self.current_date}")
            benchmark_close = None
        return benchmark_close

    def get_next_trading_day(self, date):
        while True:
            date += timedelta(days=1)
            if date > self.end_date:
                return None
            if date in self.trading_days_set:
                return date
    def get_current_date(self):
        if self.current_date is not None and self.current_date <= self.end_date:
            if self.bool:
                self.bool = False
            else:
                self.current_date = self.get_next_trading_day(self.current_date)
            return self.current_date
        else:
            return None

    def get_next_trading_date(self, current_date):
        try:
            # 获取当前日期在交易日列表中的索引位置
            index = self.trading_days.get_loc(current_date)
            if index + 1 < len(self.trading_days):
                # 返回下一个交易日的日期
                return self.trading_days[index + 1]
            else:
                # 如果当前日期已经是最后一个交易日，返回 None
                return None
        except KeyError:
            # 如果当前日期不在交易日列表中，打印错误信息并返回 None
            print(f"Current date {current_date} not found in trading days.")
            return None

    def get_previous_trading_date(self, current_date):

        try:
            index = self.trading_days.get_loc(current_date)
            if index > 0:
                previous_date = self.trading_days[index - 1]
                return previous_date
            else:
                logging.warning(f"No previous trading day available for {current_date}")
                return None
        except KeyError:
            logging.error(f"Date {current_date} not found in trading days.")
            return None

    def get_dividend_event(self, date):
        """处理股票分红事件
        :param date: 日期
        """
        try:
            import akshare as ak  
        except ImportError:
            raise ImportError(
                "请安装 akshare 以使用自动分红数据获取功能: "
            )
        date_str = date.strftime('%Y%m%d')
        try:
            #获取股票分红事件信息
            dividend_baidu_df = ak.news_trade_notify_dividend_baidu(date_str)
            if not dividend_baidu_df.empty:
                # 处理 NA 或空值
                dividend_baidu_df['分红'] = dividend_baidu_df['分红'].apply(self.extract_number)
                dividend_baidu_df['送股'] = dividend_baidu_df['送股'].apply(self.extract_dividend_and_bonus)
                dividend_baidu_df['转增'] = dividend_baidu_df['转增'].apply(self.extract_conversion_ratio)
                # 获取当前持仓股票列表
                stock_hold_now = set(self.positions.keys())
                dividend_stocks = set(dividend_baidu_df['股票代码'])
                # 只处理当前持仓且有分红信息的股票
                stocks_to_process = stock_hold_now & dividend_stocks
                for symbol in stocks_to_process:
                    # 获取该股票的分红、送股和转增信息
                    dividend_info = dividend_baidu_df.loc[dividend_baidu_df['股票代码'] == symbol].iloc[0]
                    # 获取当前持仓信息
                    position_lst = self.positions[symbol]
                    dividend_per_share = dividend_info['分红'] / 10
                    stock_dividend_ratio = dividend_info['送股'] / 10
                    capital_increase_ratio = dividend_info['转增'] / 10
                    for position in position_lst:
                        # 更新可用资金
                        dividend_amount = round(position['shares'] * dividend_per_share,2)
                        self.current_capital += dividend_amount
                        logging.info(f"{date}: 收到 {symbol} 的分红 {dividend_amount}.")
                        # 更新持仓数量
                        new_shares = round(position['shares'] * (1 + (stock_dividend_ratio + capital_increase_ratio)),0)
                        logging.info(f"{date}: {symbol} 发生送股转增，新增 {new_shares - position['shares']} 股.")
                        # 更新成本基础
                        new_cost = (position['cost'] * position['shares'] / new_shares) if new_shares > 0 else 0

                        # 更新持仓信息
                        position['shares'] = new_shares
                        position['cost'] = new_cost
                return self.positions
        except Exception as e:
            logging.error(f"Error processing dividend event on {date}: {e}")
            return self.positions

    def update_position(self, symbol, quantity, trade_price, trade_date):
        """更新持仓信息，记录每次买入的详细信息
        :param symbol: 股票代码
        :param quantity: 交易数量
        :param trade_price: 交易价格
        :param trade_date: 交易日期
        """
        if quantity == 0:
            return
        if symbol in self.positions:
            position_lst = self.positions[symbol]
            if quantity > 0:
                position_lst.append({
                    'shares': quantity,
                    'cost': trade_price,
                    'purchase_date': trade_date
                })
            else:
                # 处理卖出操作，减少持仓数量
                remaining_quantity = -quantity  # 将卖出数量转换为正数
                position_shares = sum(
                    position['shares'] for position in position_lst if position['purchase_date'] != trade_date)
                if position_shares < remaining_quantity:
                    logging.warning(f"Insufficient position quantity for {symbol}")
                    return
                for i in range(len(position_lst)):
                    record = position_lst[i]
                    if record['shares'] >= remaining_quantity:
                        record['shares'] -= remaining_quantity
                        if record['shares'] == 0:
                            del position_lst[i]  # 删除已完全卖出的记录
                        break  # 完成卖出，退出循环
                    else:
                        remaining_quantity -= record['shares']
                        record['shares'] = 0  # 标记为已卖出

                position = [record for record in position_lst if record['shares'] > 0]
                if not position:
                    del self.positions[symbol]

        else:
            if quantity > 0:
                self.positions[symbol] = [{
                    'shares': quantity,
                    'cost': trade_price,
                    'purchase_date': trade_date
                }]
            else:
                logging.warning(f"No position found for {symbol}")

    def buy_trade(self, symbol, quantity, price_type):
        """执行买入交易
            :param symbol: 股票代码
            :param quantity: 买入数量
            :param price_type: 价格类型，'open' 表示开盘价，'close' 表示收盘价，'limit' 表示指定价格
            :return: 是否成功买入
        """
        if quantity <= 0:
            logging.warning(f"Invalid quantity: {quantity}")
            raise ValueError(f"Invalid quantity: {quantity}, must be positive")

        current_date = self.current_date
        next_date = self.get_next_trading_date(current_date)
        if next_date is None:
            logging.warning("No more trading dates available.")
            raise
        try:
            trade_price = self.get_buy_trade_price(symbol, price_type, current_date, next_date)
            if trade_price is None:
                return False
            total_cost = round(trade_price * quantity * (1 + self.buy_commission),2)  # 加上交易成本
            if self.current_capital >= total_cost:
                self.current_capital = round(self.current_capital - total_cost, 2)
                self.update_position(symbol, quantity, trade_price, next_date)
                self.trades.append({
                    'date': next_date,
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': trade_price,
                    'type': 'buy'
                })
                self.trade_status[next_date].append(
                        {
                            'symbol': symbol,
                            'quantity': quantity,
                            'price': trade_price,
                            'type': 'buy'
                        }
                )
                logging.info(f"{next_date}: 买入 {symbol} {quantity} 股 at {trade_price}, 成本 {total_cost}")
                return True
            else:
                logging.warning(f"Insufficient cash to execute the trade to buy {symbol} on {next_date}")
                return False
        except Exception as e:
            logging.error(f"Error executing buy trade for {symbol}: {e}")
            return False

    def sell_trade(self, symbol, quantity, price_type):
        """执行卖出交易
        :param symbol: 股票代码
        :param quantity: 卖出数量
        :param price_type: 价格类型，'open' 表示开盘价，'close' 表示收盘价，'limit' 表示指定价格
        :return: 是否成功卖出
        """
        if quantity <= 0:
            logging.info(f"Invalid quantity: {quantity}")
            raise
        current_date = self.current_date
        next_date = self.get_next_trading_date(current_date)
        if next_date is None:
            logging.warning("No more trading dates available.")
            raise
        try:
            trade_price = self.get_sell_trade_price(symbol, price_type, current_date, next_date)
            if trade_price is None:
                return False
            current_position = self.positions.get(symbol, [])
            current_position_shares = sum(position['shares'] for position in current_position if position['purchase_date'] != next_date)
            if current_position_shares < quantity:
                logging.warning(f"Insufficient shares to execute the trade to sell {symbol} on {next_date}")
                return False
            sold_shares = 0
            total_proceeds = 0
            total_cost = 0
            for position in current_position:
                if position['purchase_date'] == next_date:
                    continue  # 跳过当天买入的股票
                if sold_shares + position['shares'] <= quantity:
                    sold_shares += position['shares']#记录已经卖出的股票数量
                    total_proceeds = round(total_proceeds + trade_price * position['shares'] * (1 - self.sell_commission),2)  # 减去交易成本
                    total_cost = round(total_cost + position['cost'] * position['shares'] * (1 + self.buy_commission),2)
                else:
                    remaining_quantity = quantity - sold_shares
                    total_proceeds = round(total_proceeds + trade_price * remaining_quantity * (1 - self.sell_commission),
                                       2)  # 减去交易成本
                    total_cost = round(total_cost + position['cost'] * remaining_quantity * (1 + self.buy_commission), 2)
                    break

            profit_loss = total_proceeds - total_cost
            self.update_position(symbol, -quantity, trade_price, next_date)
            self.current_capital = round(self.current_capital + total_proceeds, 2)
            self.trades.append({
                'date': next_date,
                'symbol': symbol,
                'quantity': -quantity,
                'price': trade_price,
                'type': 'sell',
                'profit': profit_loss
            })
            self.trade_status[next_date].append({
                'symbol': symbol,
                'quantity': -quantity,
                'price': trade_price,
                'type': 'sell'
            })

            logging.info(
                f"{next_date}: 卖出 {symbol} {quantity} 股 at {trade_price}, 收益 {total_proceeds}, 盈亏 {profit_loss}")
            return True

        except Exception as e:
            logging.error(f"Error executing sell trade for {symbol}: {e}")
            return False

    def trade_quantity(self, symbol, quantity, price_type):
        """执行买卖交易（按数量）
        :param symbol: 股票代码
        :param quantity: 目标持仓数量
        :param price_type: 价格类型，'open' 表示当日开盘价，'close' 表示前一天收盘价，'limit' 表示指定价格
        :return: 是否成功交易
        """
        if quantity < 0:
            logging.info(f"Invalid quantity: {quantity}")
            raise
        current_date = self.current_date
        next_date = self.get_next_trading_date(current_date)
        if next_date is None:
            print("No more trading dates available.")
            raise
        available_cash = self.current_capital
        current_position = self.positions.get(symbol, [])

        current_position_shares = sum(
                position['shares'] for position in current_position if position['purchase_date'] != next_date)
        trade_shares = quantity - current_position_shares
        if trade_shares == 0:
            return
        try:
            if trade_shares > 0:#执行买入操作
                trade_price = self.get_buy_trade_price(symbol, price_type, current_date, next_date)
                if trade_price is None:
                    return False
                total_cost = round(trade_price * trade_shares * (1 + self.buy_commission), 2)  # 加上交易成本
                if available_cash >= total_cost:
                    self.current_capital = round(self.current_capital - total_cost, 2)
                    self.update_position(symbol, trade_shares, trade_price, next_date)
                    self.trades.append({
                        'date': next_date,
                        'symbol': symbol,
                        'quantity': trade_shares,
                        'price': trade_price,
                        'type': 'buy'
                    })
                    self.trade_status[next_date].append({
                        'symbol': symbol,
                        'quantity': trade_shares,
                        'price': trade_price,
                        'type': 'buy'
                    })
                    logging.info(f"{next_date}: 买入 {symbol} {trade_shares} 股 at {trade_price}, 成本 {total_cost}")
                    return True
                else:
                    logging.warning(f"Insufficient cash to execute the trade to buy {symbol} on {next_date}")
                    return False

            if trade_shares < 0:#执行卖出操作
                trade_price = self.get_sell_trade_price(symbol, price_type, current_date, next_date)
                if trade_price is None:
                    return False
                total_proceeds = 0
                total_cost = 0
                sold_quantity = 0
                for position in current_position:
                    if position['purchase_date'] == next_date:
                        continue  # 跳过当天买入的股票
                    if sold_quantity + position['shares'] <= abs(trade_shares):
                        sold_quantity += position['shares']# 标记已经卖出的股票数量
                        total_proceeds += round(trade_price * position['shares'] * (1 - self.sell_commission), 2)  # 减去交易成本
                        total_cost += round(position['cost'] * position['shares'] * (1 + self.buy_commission), 2)
                    else:
                        remaining_quantity = abs(trade_shares) - sold_quantity
                        total_proceeds += round(trade_price * remaining_quantity * (1 - self.sell_commission), 2)  # 减去交易成本
                        total_cost += round(position['cost'] * remaining_quantity * (1 + self.buy_commission), 2)
                        break
                profit_loss = total_proceeds - total_cost
                self.update_position(symbol, trade_shares, trade_price, next_date)
                self.current_capital = round(self.current_capital + total_proceeds, 2)
                self.trades.append({
                    'date': next_date,
                    'symbol': symbol,
                    'quantity': trade_shares,
                    'price': trade_price,
                    'type': 'sell',
                    'profit': profit_loss
                })
                self.trade_status[next_date].append({
                    'symbol': symbol,
                    'quantity': trade_shares,
                    'price': trade_price,
                    'type': 'sell'
                })
                logging.info(
                    f"{next_date}: 卖出 {symbol} {trade_shares} 股 at {trade_price}, 收益 {total_proceeds}, 盈亏 {profit_loss}")
                return True
        except Exception as e:
            logging.error(f"Error executing trade for {symbol}: {e}")
            return False

    def trade_weight(self, symbol, weight, price_type):
        """执行买卖交易（按权重）
            :param symbol: 股票代码
            :param weight: 按可用资金分配的权重（0 到 1 之间）,weight=0表示清仓股票
            :param price_type: 价格类型，'open' 表示当日开盘价，'close' 表示前一天收盘价，'limit' 表示指定价格
            :return: 是否成功交易
        """
        # 检查 price_type 是否为字符串或数值
        if not (isinstance(price_type, str) or isinstance(price_type, (int, float))):
            logging.error(f"Invalid price_type: {price_type}. Expected a string or numeric value.")
            raise

        current_date = self.current_date
        next_date = self.get_next_trading_date(current_date)

        if next_date is None:
            logging.warning("No more trading dates available.")
            raise

        if weight < 0:
            logging.warning(f"Invalid quantity for {symbol}: {weight}")
            raise
        if weight == 0:
            current_position = self.positions.get(symbol, [])
            if not current_position:
                logging.warning('{symbol} has no shares to sell')
                return False
            # 计算当前持仓中不包括当天买入的股票
            current_position_shares = sum(
                position['shares'] for position in current_position if position['purchase_date'] != next_date)
            # 卖出全部持仓
            trade_price = self.get_sell_trade_price(symbol, price_type, current_date, next_date)
            if trade_price is None:
                return False
            total_proceeds = 0
            total_cost = 0

            for position in current_position:
                if position['purchase_date'] == next_date:
                    continue  # 跳过当天买入的股票
                total_proceeds += round(trade_price * position['shares'] * (1 - self.sell_commission), 2)  # 减去交易成本
                total_cost += round(position['cost'] * position['shares'] * (1 + self.buy_commission), 2)  # 加上交易成本
            self.update_position(symbol, -current_position_shares, trade_price, next_date)
            self.current_capital = round(self.current_capital + total_proceeds, 2)
            # 计算盈亏
            profit_loss = round(total_proceeds - total_cost, 2)
            self.trades.append({
                'date': next_date,
                'symbol': symbol,
                'quantity': -current_position_shares,
                'price': trade_price,
                'type': 'sell',
                'profit': profit_loss
            })
            self.trade_status[next_date].append({
                'symbol': symbol,
                'quantity': -current_position_shares,
                'price': trade_price,
                'type': 'sell'
            })
            logging.info(
                f"{next_date}: 卖出 {symbol} {-current_position_shares} 股 at {trade_price}, 收益 {total_proceeds}, 盈亏 {profit_loss}")
            return True

        if weight > 0:# 买入操作
            target_value = self.current_capital * weight
            trade_price = self.get_buy_trade_price(symbol, price_type, current_date, next_date)
            if trade_price is None:
                return False

            available_cash = self.current_capital
            target_quantity = int(target_value // (trade_price*(1 + self.buy_commission)))
            total_cost = round(trade_price * target_quantity * (1 + self.buy_commission), 2)  # 加上交易成本

            if available_cash >= total_cost:
                self.current_capital = round(self.current_capital - total_cost, 2)
                self.update_position(symbol, target_quantity, trade_price, next_date)
                self.trades.append({
                    'date': next_date,
                    'symbol': symbol,
                    'quantity': target_quantity,
                    'price': trade_price,
                    'type': 'buy'
                })
                self.trade_status[next_date].append({
                    'symbol': symbol,
                    'quantity': target_quantity,
                    'price': trade_price,
                    'type': 'buy'
                })
                logging.info(f"{next_date}: 买入 {symbol} {target_quantity} 股 at {trade_price}, 成本 {total_cost}")
                return True
            else:
                logging.warning(f"Insufficient cash to execute the trade to buy {symbol} on {next_date}")
                return False

    def trade_total_weight(self, symbol, weight, price_type):
        """执行买卖交易（按权重）
            :param symbol: 股票代码
            :param weight: 按总资金分配的权重（0 到 1 之间）,weight=0表示清仓股票
            :param price_type: 价格类型，'open' 表示当日开盘价，'close' 表示前一天收盘价，'limit' 表示指定价格
            :return: 是否成功交易
        """
        if weight < 0:
            error_message = f"Invalid weight for {symbol}: {weight}. Weight cannot be negative."
            logging.error(error_message)
            raise ValueError(error_message)
        # 检查 price_type 是否为字符串或数值
        if not (isinstance(price_type, str) or isinstance(price_type, (int, float))):
            error_message = f"Invalid price_type: {price_type}. Expected a string or numeric value."
            logging.error(error_message)
            raise ValueError(error_message)
        current_date = self.current_date
        next_date = self.get_next_trading_date(current_date)
        if next_date is None:
            logging.warning("No more trading dates available.")
            raise StopIteration(f"No more trading dates available after {current_date}")

        total_value = self.calculate_portfolio_value(current_date)
        last_close_price = self.get_close(symbol, current_date)
        if last_close_price is None:
            error_message = (f"No close price available for {symbol} on {current_date}. "
                             "This indicates a potential issue with the code logic or data source.")
            logging.error(error_message)
            raise ValueError(error_message)
        target_value = total_value * weight
        target_shares = int(target_value//(last_close_price*(1 + self.buy_commission)))
        position_lst = self.positions.get(symbol, [])
        current_position_shares = sum(position['shares'] for position in position_lst)
        trade_shares = target_shares - current_position_shares
        if trade_shares == 0:
            return
        if trade_shares > 0:#执行买入操作
            trade_price = self.get_buy_trade_price(symbol, price_type, current_date, next_date)
            if trade_price is None:
                return False
            available_cash = self.current_capital
            total_cost = round(trade_price * trade_shares * (1 + self.buy_commission),2)  # 加上交易成本
            if available_cash >= total_cost:
                self.current_capital -= total_cost
                self.update_position(symbol, trade_shares, trade_price, next_date)
                self.trades.append({
                    'date': next_date,
                    'symbol': symbol,
                    'quantity': trade_shares,
                    'price': trade_price,
                    'type': 'buy'
                })
                self.trade_status[next_date].append({
                    'symbol': symbol,
                    'quantity': trade_shares,
                    'price': trade_price,
                    'type': 'buy'
                })
                logging.warning(f"{next_date}: 买入 {symbol} {trade_shares} 股 at {trade_price}, 成本 {total_cost}")
                return True
            else:
                logging.warning(f"Insufficient cash to execute the trade to buy {symbol} on {next_date}")
                return False

        if trade_shares < 0:#执行卖出操作
            trade_price = self.get_sell_trade_price(symbol, price_type, current_date, next_date)
            if trade_price is None:
                return False
            total_proceeds = 0
            total_cost = 0
            sold_quantity = 0
            for position in position_lst:
                if position['purchase_date'] == next_date:
                    continue
                if sold_quantity + position['shares'] <= abs(trade_shares):
                    sold_quantity += position['shares']  # 标记已经卖出的股票数量
                    total_proceeds += round(trade_price * position['shares'] * (1 - self.sell_commission),2)  # 减去交易成本
                    total_cost += round(position['cost'] * position['shares'] * (1 + self.buy_commission), 2)
                else:
                    remaining_quantity = abs(trade_shares) - sold_quantity
                    sold_quantity += remaining_quantity
                    total_proceeds += round(trade_price * remaining_quantity * (1 - self.sell_commission),2)  # 减去交易成本
                    total_cost += round(position['cost'] * remaining_quantity * (1 + self.buy_commission), 2)
                    break
            if sold_quantity < abs(trade_shares):
                raise(f'Insufficient shares to execute the trade to sell {symbol} {trade_shares}on {next_date}')
            self.update_position(symbol, trade_shares, trade_price, next_date)
            self.current_capital = round(self.current_capital + total_proceeds, 2)
            profit_loss = round(total_proceeds - total_cost, 2)
            self.trades.append({
                'date': next_date,
                'symbol': symbol,
                'quantity': trade_shares,
                'price': trade_price,
                'type': 'sell',
                'profit': profit_loss
            })
            self.trade_status[next_date].append({
                'symbol': symbol,
                'quantity': trade_shares,
                'price': trade_price,
                'type': 'sell'
            })
            logging.info(
                    f"{next_date}: 卖出 {symbol} {trade_shares} 股 at {trade_price}, 收益 {total_proceeds}, 盈亏 {profit_loss}")
            return True

    def get_trade_price(self, symbol, price_type, current_date, next_date, is_buy: bool):
        """通用方法获取买入或卖出交易价格
        :param symbol: 股票代码
        :param price_type: 价格类型
        :param current_date: 当前日期
        :param next_date: 下一个交易日期
        :param is_buy: 是否为买入操作
        :return: 交易价格或 None
        """
        open_price = self.get_open(symbol, next_date)
        if open_price is None:
            logging.warning(f"No open price available for {symbol} on {next_date}.")
            return None
        if price_type == 'open':
            trade_price = open_price
        elif price_type == 'close':
            last_close_price = self.get_close(symbol, current_date)
            if last_close_price is None:
                logging.warning(f"No close price available for {symbol} on {current_date}.")
                return None
            if is_buy:
                low_next_date = self.get_low(symbol, next_date)
                if low_next_date is None:
                    logging.warning(f"No low price available for {symbol} on {next_date}.")
                    return None
                if open_price <= last_close_price:
                    trade_price = open_price
                elif low_next_date < last_close_price:
                    trade_price = last_close_price
                else:
                    logging.warning(
                        f"{next_date} buy {symbol} Trade not executed as the next day's low price ({low_next_date}) is higher than the last close price ({last_close_price}).")
                    return None
            else:
                high_next_date = self.get_high(symbol, next_date)
                if high_next_date is None:
                    logging.warning(f"No high price available for {symbol} on {next_date}.")
                    return None
                if open_price >= last_close_price:
                    trade_price = open_price
                elif high_next_date > last_close_price:
                    trade_price = last_close_price
                else:
                    logging.warning(
                        f"{next_date} sell {symbol} Trade not executed as the next day's high price ({high_next_date}) is lower than the last close price ({last_close_price}).")
                    return None

        elif isinstance(price_type, (int, float)):
            limit_price = price_type
            if is_buy:
                if open_price <= limit_price:
                    trade_price = open_price
                else:
                    low_next_date = self.get_low(symbol, next_date)
                    if low_next_date is None or low_next_date >= limit_price:
                        logging.warning(
                            f"{next_date} buy {symbol} Trade not executed as the next day's low price ({low_next_date}) is higher than the limit price ({limit_price}).")
                        return None
                    trade_price = limit_price
            else:
                if open_price >= limit_price:
                    trade_price = open_price
                else:
                    high_next_date = self.get_high(symbol, next_date)
                    if high_next_date is None or high_next_date <= limit_price:
                        logging.warning(
                            f"{next_date} sell {symbol} Trade not executed as the next day's high price ({high_next_date}) is lower than the limit price ({limit_price}).")
                        return None
                    trade_price = limit_price
        else:
            logging.warning("Invalid price type.")
            return None
        return trade_price

    def get_buy_trade_price(self, symbol, price_type, current_date, next_date) -> float:
        return self.get_trade_price(symbol, price_type, current_date, next_date, is_buy=True)

    def get_sell_trade_price(self, symbol, price_type, current_date, next_date) -> float:
        return self.get_trade_price(symbol, price_type, current_date, next_date, is_buy=False)

    def get_previous_nth_trading_day_from(self, date, position_days):
        """从 date 开始，向前找到第 position_days 个交易日"""
        # 找到 date 在 self.trading_days 中的索引位置
        trading_days_array = self.trading_days.to_numpy()
        # 确保 date 是 numpy.datetime64 类型
        date_np = np.datetime64(date)
        idx = trading_days_array.searchsorted(date_np)
        if idx < 0:
            raise IndexError("The given date is earlier than the earliest trading day.")
        # 计算新的索引位置
        new_idx = idx - position_days
        # 检查新索引是否超出范围
        if new_idx < 0:
            logging.info("The calculated index is out of bounds (less than 0).")
            new_idx = 0
        # 获取新的日期
        nth_trading_day = self.trading_days[new_idx]
        return nth_trading_day

    def trade_position_days(self):
        """根据持仓天数进行交易"""
        if not self.positions:
            return False
        next_date = self.get_next_trading_date(self.current_date)
        if not next_date:
            logging.info(f'No more next_date')
            raise

        positions_copy = self.positions.copy()
        target_date = self.get_previous_nth_trading_day_from(self.current_date,self.position_days)

        for stock,position_lst in positions_copy.items():
            trade_shares = sum(
                position['shares']
                for position in position_lst
                if position['purchase_date'] < target_date
            )
            if trade_shares == 0:
                continue
            trade_price = self.get_open(stock, next_date)
            logging.warning(
                f"Days triggered for {stock} at {next_date}, selling {trade_shares} shares")
            if trade_price is None:
                logging.warning(f"No open price available for {stock} on {next_date}.")
                continue
            total_proceeds = round(trade_price * trade_shares * (1 - self.sell_commission), 2)
            position_cost = sum(i['cost'] * i['shares'] * (1 + self.buy_commission)
                                for i in position_lst
                                if i['purchase_date'] < target_date
                                )
            profit_loss = round(total_proceeds - position_cost, 2)
            self.update_position(stock, -trade_shares, trade_price, next_date)
            self.current_capital = round(self.current_capital + total_proceeds, 2)
            self.trades.append({
                'date': next_date,
                'symbol': stock,
                'quantity': -trade_shares,
                'price': trade_price,
                'type': 'sell',
                'profit': profit_loss
            })
            self.trade_status[next_date].append({
                'symbol': stock,
                'quantity': -trade_shares,
                'price': trade_price,
                'type': 'sell'
            })
            logging.info(
                f"{next_date}: 卖出 {stock} {-trade_shares} 股 at {trade_price}, 收益 {total_proceeds}, 盈亏 {profit_loss}")
        return True

    def stop_loss(self):
        """执行止损操作"""
        if not self.positions:
            logging.warning("No positions to process.")
            return
        positions_copy = self.positions.copy()
        for stock, position_records in positions_copy.items():
            for position in position_records:
                cost_price = position.get('cost', 0)
                if cost_price == 0:
                    logging.warning(f"Warning: No 'cost' key found for stock {stock}")
                    continue
                try:
                    close_price = self.get_close(stock, self.current_date)
                    if close_price is None:
                        logging.warning(f"Failed to get close price for {stock} on {self.current_date}")
                        continue
                    cost_return = (close_price / cost_price - 1) * 100
                    next_date = self.get_next_trading_date(self.current_date)
                    if cost_return <= self.stop_loss_return and position['purchase_date'] != next_date:
                        shares = position['shares']
                        trade_price = self.get_open(stock, next_date)
                        logging.warning(
                            f"Stop loss triggered for {stock} at {self.current_date}, selling {shares} shares")
                        if trade_price is None:
                            logging.warning(f"No open price available for {stock} on {next_date}.")
                            continue
                        total_proceeds = round(trade_price * shares * (1 - self.sell_commission), 2)
                        profit_loss = round(
                            total_proceeds - position['cost'] * position['shares'] * (1 + self.buy_commission), 2)
                        self.update_position(stock, -shares, trade_price, next_date)
                        self.current_capital = round(self.current_capital + total_proceeds,2)
                        self.trades.append({
                            'date': next_date,
                            'symbol': stock,
                            'quantity': -shares,
                            'price': trade_price,
                            'type': 'sell',
                            'profit': profit_loss
                        })
                        self.trade_status[next_date].append({
                            'symbol': stock,
                            'quantity': -shares,
                            'price': trade_price,
                            'type': 'sell'
                        })
                        logging.info(
                            f"{next_date}: 卖出 {stock} {-shares} 股 at {trade_price}, 收益 {total_proceeds}, 盈亏 {profit_loss}")
                except Exception as e:
                    logging.warning(f"Error processing stop loss for {stock}: {e}")

    def move_stop_loss(self):
        pass

    def update_results(self,date):
        # 更新交易结果记录
        self.portfolio_value = self.calculate_portfolio_value(date)
        self.results.append({
            'date': date,
            'portfolio_value': self.portfolio_value
        })

    def calculate_portfolio_value(self,date):
        """计算特定日期的投资组合价值
        :param date: 计算价值的日期
        :return: 投资组合的价值
        """
        total_value = round(self.current_capital, 2)
        self.position_value = 0

        for symbol, position_records in self.positions.items():
            # 计算每个股票的总持仓数量
            total_shares = sum(record['shares'] for record in position_records)
            if total_shares > 0:
                current_prices = self.get_close(symbol, date)
                self.position_value += round(total_shares * current_prices, 2)
        total_value += self.position_value
        return total_value

    def get_trades(self):
        return self.trades  # 返回所有交易记录

    def calculate_metrics(self):
        try:
            results_df = pd.DataFrame(self.results)
            if results_df.empty:
                print("No data available to calculate metrics.")
                return {
                    'Total Return': 0.0,
                    'Annualized Return': 0.0,
                    'Max Drawdown': 0.0,
                    'Sharpe Ratio': 0.0,
                    'win_rate': 0.0
                }
            results_df.set_index('date', inplace=True)
            pv = results_df['portfolio_value']
            # 计算总回报率
            total_return = (pv.iloc[-1] - pv.iloc[0]) / pv.iloc[0]
            logging.info(results_df['portfolio_value'].iloc[0])
            logging.info(results_df['portfolio_value'].iloc[-1])
            # 计算年化收益率
            annual_return = (1 + total_return) ** (252 / (len(pv) - 1)) - 1
            # 计算最大回撤
            rolling_max = pv.cummax()
            daily_drawdown = (pv - rolling_max) / rolling_max
            max_drawdown = daily_drawdown.min()
            # 日收益率（用于夏普）
            daily_returns = pv.pct_change().dropna()
            # 计算夏普比率
            if len(daily_returns) < 2 or daily_returns.std() == 0:
                sharpe_ratio = 0.0
            else:
                # 假设无风险利率为 0（因现金未计息）
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            # 计算成功率
            sell_trades = [t for t in self.trades if t['type'] == 'sell']
            winning_trades = [t for t in sell_trades if t['profit'] > 0]
            win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
            print(f"交易胜率: {win_rate:.2%}")
            # 返回绩效指标
            return {
                'Total Return': total_return,
                'Annualized Return': annual_return,
                'Max Drawdown': max_drawdown,
                'Sharpe Ratio': sharpe_ratio,
                'win_rate' : win_rate
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                    'Total Return': 0.0,
                    'Annualized Return': 0.0,
                    'Max Drawdown': 0.0,
                    'Sharpe Ratio': 0.0,
                    'win_rate': 0.0
                    }

    def plot_results(self):
        if not self.results:
            logging.warning("No results to plot.")
            return
        results_df = pd.DataFrame(self.results)
        results_df['date'] = pd.to_datetime(results_df['date'])
        results_df.set_index('date', inplace=True)
        # 计算起始资金（第一天的portfolio_value）
        initial_capital = results_df['portfolio_value'].iloc[0]
        # 计算投资组合价值的变化率
        results_df['portfolio_return'] = results_df['portfolio_value'] / initial_capital - 1
        # 创建图表
        plt.figure(figsize=(12, 6))
        plt.plot(results_df.index, results_df['portfolio_return'],
                 label='Portfolio Return', color='blue')

        # 绘制基准指数的时间序列图
        if hasattr(self, 'benchmark_data') and self.benchmark_data is not None:
            benchmark = self.benchmark_data.reindex(results_df.index).ffill().dropna()
            if not benchmark.empty:
                bench_init = benchmark['close'].iloc[0]
                benchmark['benchmark_return'] = benchmark['close'] / bench_init - 1
                plt.plot(benchmark.index, benchmark['benchmark_return'],
                         label='Benchmark (HS300) Return', color='orange')

        # 添加绩效指标的文本标签
        metrics = self.calculate_metrics()
        self.add_metric_text(metrics)
        # 设置图表标题和标签
        plt.title('Backtest Results: Cumulative Return')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')  # ← 关键修正！
        plt.grid(True)
        plt.legend()
        plt.tight_layout()  # 防止标签被裁剪
        plt.show()

    def add_metric_text(self, metrics):
        """在图表上添加绩效指标文本"""
        plt.text(0.05, 0.95, f"Total Return: {metrics['Total Return'] * 100:.2f}%",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.90, f"Annualized Return: {metrics['Annualized Return'] * 100:.2f}%",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.85, f"Max Drawdown: {metrics['Max Drawdown'] * 100:.2f}%",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.80, f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    def default_run(self):
        # 默认的空策略函数
        pass

    def log_current_status(self, date):
        """记录当前状态"""
        self.current_capital = round(self.current_capital,2)
        self.position_value = round(self.position_value, 2)
        self.portfolio_value = round(self.portfolio_value, 2)
        logging.info(f'{date} 持仓: {self.positions}')
        logging.info(f"{date} 当前可用资本: {self.current_capital}")
        logging.info(f"{date} 当前持仓价值: {self.position_value}")
        logging.info(f"{date} 总投资组合价值: {self.portfolio_value}")

    def log_metrics(self):
        """计算并记录绩效指标"""
        metrics = self.calculate_metrics()
        logging.info("回测绩效指标:")
        for key, value in metrics.items():
            if key == 'Sharpe Ratio':
                logging.info(f"{key}: {value:.2f}")
            else:
                logging.info(f"{key}: {value * 100:.2f}%")

    def run_backtest(self):
        while self.get_current_date() is not None:
            next_date = self.get_next_trading_date(self.current_date)
            if next_date is None:
                logging.info('No more next trading days available.')
                break
            try:
                # 运行策略函数
                self.run_func(self)
                #卖出到期的股票
                self.trade_position_days()
                # 处理分红、送股和转增事件
                if self.auto_adjust:
                    self.get_dividend_event(next_date)
                # 处理止损
                if self.stop_loss_return is not None:
                    self.stop_loss()
                # 更新回测结果
                self.update_results(next_date)
                self.log_current_status(next_date)
                # 计算并输出绩效指标
                self.log_metrics()

            except Exception as e:
                logging.error(f"Error on {self.current_date}: {e}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                continue


        # 回测完成后绘制结果
        self.plot_results()
