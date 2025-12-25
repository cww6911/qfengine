# test_demo.py

import pandas as pd
import qfengine

print("✅ 成功导入 qfengine！")
print(f"版本: {qfengine.__version__}")
print(f"函数: {qfengine.make_qfq}, {qfengine.make_hfq}")

# 创建一个最小测试数据（含一次分红）
df = pd.DataFrame({
    'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
    'symbol': ['000001', '000001', '000001'],
    'open': [10.0, 5.0, 5.2],
    'high': [10.5, 5.3, 5.4],
    'low': [9.8, 4.9, 5.0],
    'close': [10.2, 5.1, 5.3],
    '股权系数': [1.0, 2.0, 1.0],      # 第二天 10送10
    '派息系数': [0.0, 0.0, 0.0],
    '成交量': [1000, 2000, 1500],
    '换手率': [0.1, 0.2, 0.15],
    '流通股本': [10000, 20000, 20000]
})
df['date'] = pd.to_datetime(df['date'])

# 测试后复权
print("\n🔍 测试 make_qfq (后复权):")
try:
    result_qfq = df.groupby('symbol').apply(qfengine.make_qfq).reset_index(drop=True)
    print(result_qfq[['date', 'close', 'close_qfq']])
except Exception as e:
    print("❌ 后复权出错:", e)

# 测试前复权
print("\n🔍 测试 make_hfq (前复权):")
try:
    result_hfq = df.groupby('symbol').apply(qfengine.make_hfq).reset_index(drop=True)
    print(result_hfq[['date', 'close', 'close_hfq']])
except Exception as e:
    print("❌ 前复权出错:", e)

print("\n🎉 测试完成！如果看到价格列，说明你的包工作正常！")
