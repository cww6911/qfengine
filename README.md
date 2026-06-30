中文名：观复形态


英文名：MorphGuanFu




\# QFEngine — 精确复权 · 可信回测 · 智能选股，下载结构化技术形态数据



> 一个面向量化研究与实盘模拟的轻量级 Python 引擎，  提供自研的结构化技术形态数据下载。

> 核心特色：\*\*数学保真的价格复权 + 可审计的交易记录 + 股票池驱动的策略框架\*\*。



---



\## ✨ 核心特性：


# QF Engine：交易场论量化数据

> **收益不在K线里，而在场中。**  
> 本项目基于「交易场论」（Trading Field Theory）构建量化研究基技术形态数据，开源**技术形态数据集**与**策略逻辑框架**，旨在探索市场结构的本质坐标。

---

## 📊 技术形态数据集（Fractal Position Dataset）

我们提供中国全市场A股品种的**场坐标预计算数据**，将原始价格映射到具有物理意义的“场空间”中，为策略开发提供新维度。

### ✨ 核心特性
- **场坐标定义**：
  - `rate_position`
  - `pressure_position`：（基于分形嵌套结构的压力强度坐标）
  - （更多字段见数据样例）
- **覆盖范围**：A股（以后会扩展）
- **时间粒度**：日线级别，从 1995 年至2025年（以后会定时更新）
### 🗂️ 存储结构：数据按日分区存储为高效列式格式


> 💡 使用 **Git LFS** 管理大文件，确保仓库轻量可克隆。

---

## 🚀 快速开始

### 1. 克隆仓库 + 下载数据
```bash
git clone https://github.com/cww6911/qfengine.git
cd qfengine
git lfs install      # 首次使用需安装 Git LFS
git lfs pull         # 下载所有 .parquet 数据文件

📥 不想用 Git？  
→ [点击下载完整数据集（ZIP 格式）](https://github.com/cww6911/qfengine/releases)

---

加载数据（Python 示例）：

import pandas as pd

# 示例：加载全部数据
df = pd.read_parquet('data/')

# 示例：加载2025年
df_2025 = pd.read_parquet('data/', filters=[('date', '>=', '2025-01-01')])

# 查看核心字段
print(df[['symbol', 'date', 'rate_position', 'pressure_position']].head())




\- \*\*精确复权（QFQ / HFQ）\*\*  

&nbsp; 基于总回报率重建价格序列，确保每日涨跌幅 = 真实收益，杜绝传统因子法的累积误差。

&nbsp; 

\- \*\*事件驱动回测引擎\*\*  

&nbsp; 支持按日调度、动态调仓、持仓管理、绩效计算（年化收益、最大回撤、夏普比率、胜率等）。



\- \*\*股票池（Stock Pool）机制\*\*  

&nbsp; 外部注入候选股票列表，支持“信号日 → 交易日”延迟执行，贴近实盘逻辑。



\- \*\*结构化交易记录\*\*  

&nbsp; 所有成交按日期索引存储，便于事后分析与可视化。



---



\## 📦 安装



从源码安装（开发模式）：

```bash

git clone https://github.com/chenwenwai/qfengine.git

cd qfengine

pip install -e .





依赖：pandas >= 1.3, numpy >= 1.20





快速开始



1\. 数据准备

你的行情数据需包含以下字段（DataFrame）：



字段                                             说明

date                                            交易日（datetime）

symbol                                       股票代码（str）

open, high, low, close              OHLC 价格

股权系数                                      除权因子（如 10送5 → 1.5；无分红填 1）

派息系数                                      派息金额（元/股；无派息填 0）



&nbsp;分红事件需提前合并到行情表中（同日期对齐）。



2\. 计算复权价格

import pandas as pd

from qfengine import make\_qfq, make\_hfq



\# 按股票分组复权

df\_qfq = df.groupby('symbol').apply(make\_qfq).reset\_index(drop=True)    #  返回向后复权价格

df\_hfq = df.groupby('symbol').apply(make\_hfq).reset\_index(drop=True)    #  返回向前复权价格



3\. 运行回测

from qfengine.backtest import BacktestEngine

\# 用户写策略函数，context是全局对象，run\_func是函数名，是引擎规定的，必须按规定格式写

def run\_func(context):

&nbsp;	pass

\#   调用回测引擎，start\_date是回测开始日期，end\_date是回测结束日期， data是传入的数据， run\_func是策略函数。

context = BacktestEngine(start\_date, end\_date, data, run\_func)





&nbsp; 回测输出示例

Total Return: 15.88%

Annualized Return: 16.66%

Max Drawdown: -12.81%

Sharpe Ratio: 0.79

Win Rate: 20.00%



&nbsp; 项目结构

qfengine/

├── data/          # 复权模块

│   └── adjust.py  # make\_qfq, make\_hfq

├── backtest/      # 回测引擎

│   └── engine.py  # BacktestEngine 类

└── \_\_init\_\_.py    # 统一导出接口





&nbsp; 贡献与扩展

欢迎提交 Issue 或 PR！

&nbsp;当前可扩展方向：

支持多因子选股

集成 AKShare / Tushare 数据自动下载

添加净值曲线绘图功能



&nbsp; License

MIT © \[chenwenwai]





**复权数据准备说明：**

1、原始历史未复权行情数据

2、对应时间段的历史分红数据，可以从akshare获取

3、代码示例:



def merge\_price\_with\_dividend\_events(df\_price, df\_dividend):

&nbsp;   """

&nbsp;   将未复权行情数据与分红送股事件数据按「除权日」进行左连接（left join），

&nbsp;   用于后续复权价格计算。分红送股事件数据要进行必要的清洗，必须要有股票代码symbol列

&nbsp;   Parameters:

&nbsp;   ----------

&nbsp;   df\_price : pd.DataFrame

&nbsp;       未复权行情数据，必须包含列: \['date', 'symbol', 'open', 'high', 'low', 'close']

&nbsp;   df\_dividend : pd.DataFrame

&nbsp;       分红送股事件数据，必须包含列: \['date' (即除权日), 'symbol',

&nbsp;                                   '送股比例', '转增比例', '派息比例']

&nbsp;   Returns:

&nbsp;   -------

&nbsp;   pd.DataFrame

&nbsp;       合并后的数据，新增列: '股权系数', '派息系数'，无除权日的行将填充为1和0。

&nbsp;   """

&nbsp;   # 初始化处理

&nbsp;   df\_dividend.rename(columns={'除权日': 'date'}, inplace=True)

&nbsp;   df\_dividend.sort\_values(by='date', inplace=True)

&nbsp;   # 处理缺失值

&nbsp;   df\_dividend\['送股比例'] = df\_dividend\['送股比例'].fillna(0)

&nbsp;   df\_dividend\['转增比例'] = df\_dividend\['转增比例'].fillna(0)

&nbsp;   df\_dividend\['派息比例'] = df\_dividend\['派息比例'].fillna(0)



&nbsp;   # 计算复权所需系数

&nbsp;   df\_dividend\['股权系数'] = (df\_dividend\['送股比例'] + df\_dividend\['转增比例']) / 10.0 + 1.0

&nbsp;   df\_dividend\['派息系数'] = df\_dividend\['派息比例'] / 10.0



&nbsp;   # 保留必要列

&nbsp;   df\_dividend = df\_dividend\[\['date', 'symbol', '股权系数', '派息系数']].copy()



&nbsp;   # 确保 date 是 datetime

&nbsp;   df\_dividend\['date'] = pd.to\_datetime(df\_dividend\['date'])

&nbsp;   df\_price = df\_price.copy()

&nbsp;   df\_price\['date'] = pd.to\_datetime(df\_price\['date'])



&nbsp;   # 左连接：保留所有行情日，事件日不存在则填 NaN（后续 fillna）

&nbsp;   df\_merged = pd.merge(df\_price, df\_dividend, on=\['date', 'symbol'], how='left')



&nbsp;   # 填充无事件日的默认值

&nbsp;   df\_merged\['股权系数'] = df\_merged\['股权系数'].fillna(1.0)

&nbsp;   df\_merged\['派息系数'] = df\_merged\['派息系数'].fillna(0.0)



&nbsp;   return df\_merged





**回测引擎属性说明：**

.total\_capital                帐户起始资金 ，默认为1000000

.current\_capital            帐户当前可用资金

.positions                     帐户当前持仓信息

.buy\_commission        买入交易费用，默认为0.0005  

.sell\_commission         卖出交易费用，默认为0.0015

.basis\_index                  基准指数，默认为沪深300

.portfolio\_value                   当前账户总资产

.position\_value             当前账户持仓股票市值

.position\_days              股票持仓周期（交易日天数），默认为10天，到期后会自动执行卖出交易

.stock\_pool                   股票池，用户可以在股票池存入要跟踪观察的股票，是列表格式

.trade\_status                每日成交记录，用户可以通过这个属性查询任何一天的详细成交记录

.stop\_loss\_return         固定止损幅度，如设置为-20，则在股票收盘价低于买入成本价格的-20%时，执行自动卖出止损

说明：所有属性，用户都可以通过context.的方式进行重置修改





**回测引擎参数说明：**

start\_date                   回测开始日期（输入格式为字符串）

end\_date                    回测结束日期（输入格式为字符串）

data                            用户传入的回测数据，必须用统一的列名为:date，symbol，close，open，high，low，其他列名字段不作规定。

run\_func                     策略函数的函数名，用户不能使用其他函数名

auto\_adjust               声明传入的data是否要计算复权收益，默认为不计算，用户可以context.aute\_adjust的方式重置声明，建议使用复权数据

context                       策略函数要传入的参数，为全局对象，建议用户统一使用context这个名字



**调用交易指令说明：**

**1、context.buy\_trade(symbol, quantity, price\_type)，symbol：要买入的股票代码，quantity：买入数量（股），price\_type：买入的价格类型，分close,open,price(如10.5)**

**2、context.sell\_trade( symbol, quantity, price\_type)，quantity：卖出股票的数量，输入为正数**

**3、context.trade\_quantity(symbol, quantity, price\_type)：按目标持仓数量执行交易，quantity为目标持仓数量，如果输入0，则对该股票进行清仓**

**4、context.trade\_weight( symbol, weight, price\_type)：按可用资金分配的权重（0 到 1 之间）执行交易，weight=0表示清仓股票**

**5、context.trade\_total\_weight( symbol, weight, price\_type)：按总资金分配权重（0 到 1 之间）,weight=0表示清仓股票**



**其他说明 ：**

1、由于计算复权收益要用到akshare的数据接口，有时akshare的数据接口不稳定，分红数据如果获取失败，会导致回测结果失真，所以建议用户用复权数据进行回测。

2、用户在写策略函数时，可以用任意数据进行合并和选股逻辑计算，但传入的参数data必须符合上面的规定，要包含date，symbol，close，open，high，low这6列的数据，其他数据可以不传入。



**回测函数代码示例：**

    **def run\_func(context):**

	**“用户可以用自己的任意数据写策略逻辑”**

        **context.stop\_loss\_return = -10  # 声明固定止损幅度**

        **context.position\_days = 120  # 声明持仓周期**

        **try:**

            **current\_date = context.current\_date  #    获取当前日期，这句代码是固定格式，必须要有**



            **trades = context.trade\_status # 获取成交记录信息**

            **daily\_trades = trades.get(current\_date,\[])   #  通过trades可以获取任意日期的成交记录信息**



            **today\_df = context.data\[context.data.date == current\_date] # 这句代码是固定格式，必须要有，目的是把全时段数据转换为回测当日的数据** 



            **# 打印调试信息**

            **print(current\_date.strftime('%Y-%m-%d'))**

            **print('今天的成交股票信息', daily\_trades)**

            **# 通过positions对象，使用列表生成式的方法获取目前持仓的股票列表**

            **stock\_hold\_now = list(context.positions.keys())**

            **print(f'目前持仓的股票:{stock\_hold\_now}')**

            **if not today\_df.empty:**

                **cond = today\_df\['buy\_price'].notna()  # 选 股条件**

                **today\_df = today\_df\[cond]**

                **# 整理出当天要买入的股票**

                **stock\_to\_buy = today\_df.symbol.tolist()**



                **# 等权重买入**

                **if len(stock\_to\_buy)>10:**

                    **weight = 0.99/len(stock\_to\_buy)**

                **else:**

                    **weight = 0.1**

                **# 买入**

                **print(f'需要买入出的股票:总共{len(stock\_to\_buy)}个股票,{stock\_to\_buy}')**

                **for stock in stock\_to\_buy:**

                    **context.trade\_total\_weight(stock,weight,'close')  # 执行交易指令**



        **except Exception as e:**

            **print(f"Error in run\_func: {e}")**

实例化对象，执行回测：

context = BacktestEngine(start_date, end_date, data, run_func, auto_adjust =False)
其中参数auto_adjust也可以直接在策略函数中通过context. auto_adjust的方式进行声明修改




