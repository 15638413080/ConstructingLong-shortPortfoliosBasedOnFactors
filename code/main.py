import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


# ==================== 步骤1：正确读取数据 ====================
def read_data_correctly(filepath):
    """
    简单版本：直接使用header=1读取
    """
    try:
        df = pd.read_excel(filepath, sheet_name='sheet1', header=0, skiprows=[1, 2])
        
        print(f"✓ 成功读取数据（使用英文表头）")
        print(f"  数据形状: {df.shape}")
        
        # 查看列名
        print(f"  原始列名: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        print(f"读取失败: {e}")
        return None


# 读取数据
print("="*80)
print("开始读取数据...")
df = read_data_correctly('TRD_Dalyr.xlsx')

if df is None:
    print("数据读取失败，请检查文件路径和格式")
    exit()

print("\n" + "="*80)
print("数据基本信息：")
print(f"数据形状：{df.shape}")
print(f"数据类型统计:{df.dtypes}")
print(f"列名：{df.columns.tolist()}")
print(f"数据期间：{df['Trddt'].min()} 到 {df['Trddt'].max()}")
print(f"股票数量：{df['Stkcd'].nunique()}")
print(f"总记录数：{len(df)}")
print("\n前5行数据：")
print(df.head())

# ==================== 步骤2：数据预处理 ====================
# 确保数据按股票代码和日期排序
df = df.sort_values(['Stkcd', 'Trddt']).reset_index(drop=True)

# 添加年月标记，用于月度分组
df['year_month'] = pd.to_datetime(df['Trddt'], errors='coerce').dt.to_period('M')

print("\n" + "="*80)
print("数据预处理完成！")

# ==================== 步骤3：MAX因子计算 ====================
def process_all_stocks_max_factor(df):
    """处理所有股票的MAX因子计算"""
    
    # 数据清洗和预处理
    df = df.sort_values(['Stkcd', 'Trddt'])
    df['year_month'] = pd.to_datetime(df['Trddt'], errors='coerce').dt.to_period('M')
    
    # 检查是否有所需的收益率列
    if 'Dretwd' not in df.columns:
        print("错误：数据中缺少Dretwd列（考虑现金红利再投资的日个股回报率）")
        print("可用列：", df.columns.tolist())
        return df
    
    # 计算每只股票的MAX因子（过去20日最高收益率）
    def calculate_max_for_stock(group):
        group = group.sort_values('Trddt')
        # 使用过去20个交易日，至少需要10个交易日的数据
        group['MAX'] = group['Dretwd'].rolling(window=20, min_periods=10).max()
        return group
    
    print("正在计算MAX因子...")
    df_with_max = df.groupby('Stkcd', group_keys=False).apply(calculate_max_for_stock)
    df_with_max = df_with_max.reset_index(drop=True)
    
    # 检查MAX因子计算结果
    print(f"MAX因子计算完成，{df_with_max['MAX'].notna().sum()}个有效值")
    print(f"股票数量：{df_with_max['Stkcd'].nunique()}")
    
    return df_with_max

# 处理数据
print("\n" + "="*80)
print("开始计算MAX因子...")
df_processed = process_all_stocks_max_factor(df)

# 查看股票数量和数据范围
unique_stocks = df_processed['Stkcd'].nunique()
print(f"股票数量：{unique_stocks}")
print(f"时间范围：{df_processed['Trddt'].min()} 到 {df_processed['Trddt'].max()}")
print(f"总记录数：{len(df_processed)}")

# ==================== 步骤4：月度多空组合构建 ====================
def construct_monthly_max_portfolios_daily(df):
    """基于日度数据构建月度多空组合"""
    
    # 获取每月第一个交易日
    df['year_month'] = pd.to_datetime(df['Trddt'], errors='coerce').dt.to_period('M')
    monthly_starts = df.groupby('year_month')['Trddt'].min().reset_index()
    monthly_starts.columns = ['year_month', 'month_start']
    
    portfolio_results = []
    
    all_months = sorted(df['year_month'].unique())
    
    print(f"\n共有 {len(all_months)} 个月份数据")
    
    for i in range(1, len(all_months)):
        current_month = all_months[i-1]  # 用于分组的月份
        next_month = all_months[i]       # 计算收益的月份
        
        # 获取当前月末的MAX因子值（用于分组）
        month_end = df[df['year_month'] == current_month].groupby('Stkcd').last().reset_index()
        month_end = month_end[['Stkcd', 'MAX']].dropna()
        
        if len(month_end) < 50:  # 确保有足够股票
            print(f"月份 {current_month}: 只有 {len(month_end)} 只股票，跳过")
            continue
            
        # 按MAX值分成5组
        try:
            month_end['quantile'] = pd.qcut(month_end['MAX'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except Exception as e:
            print(f"月份 {current_month}: 分组失败 - {e}")
            continue
        
        # 获取下个月的日收益率数据
        next_month_data = df[df['year_month'] == next_month]
        
        if len(next_month_data) == 0:
            print(f"月份 {next_month}: 无数据，跳过")
            continue
            
        # 计算各组的日收益率
        daily_returns = []
        for date in sorted(next_month_data['Trddt'].unique()):
            date_data = next_month_data[next_month_data['Trddt'] == date]
            merged_data = pd.merge(date_data, month_end[['Stkcd', 'quantile']], on='Stkcd', how='inner')
            
            if len(merged_data) > 0:
                group_returns = merged_data.groupby('quantile')['Dretwd'].mean()
                for q, ret in group_returns.items():
                    daily_returns.append({
                        'date': date,
                        'quantile': q,
                        'return': ret
                    })
        
        daily_returns_df = pd.DataFrame(daily_returns)
        
        if len(daily_returns_df) == 0:
            print(f"月份 {next_month}: 无有效收益率数据，跳过")
            continue
            
        # 计算月度收益率（日收益率累乘）
        monthly_results = {}
        for q in [1, 5]:  # 只需要第1组和第5组
            q_returns = daily_returns_df[daily_returns_df['quantile'] == q]['return']
            if len(q_returns) > 0:
                monthly_return = (1 + q_returns).prod() - 1
                monthly_results[f'Q{q}'] = monthly_return
            else:
                monthly_results[f'Q{q}'] = 0
        
        # 多空组合收益
        long_short_return = monthly_results.get('Q5', 0) - monthly_results.get('Q1', 0)
        
        portfolio_results.append({
            'year_month': next_month,
            'Q1_return': monthly_results.get('Q1', 0),
            'Q5_return': monthly_results.get('Q5', 0),
            'long_short_return': long_short_return,
            'n_stocks': len(month_end),
            'n_trading_days': len(daily_returns_df['date'].unique())
        })
        
        if len(portfolio_results) % 12 == 0:
            print(f"已处理 {len(portfolio_results)} 个月份")
    
    return pd.DataFrame(portfolio_results)

# 构建多空组合
print("\n" + "="*80)
print("开始构建月度多空组合...")
portfolio_results = construct_monthly_max_portfolios_daily(df_processed)

print(f"\n成功构建 {len(portfolio_results)} 个月的组合")

if len(portfolio_results) == 0:
    print("错误：未能构建任何组合，请检查数据")
    exit()

# ==================== 步骤5：绩效指标计算 ====================
def calculate_performance_metrics(returns_series, periods_per_year=12):
    """计算绩效指标"""
    if len(returns_series) == 0:
        return np.nan, np.nan, np.nan
    
    # 移除NaN值
    returns_clean = returns_series.dropna()
    if len(returns_clean) == 0:
        return np.nan, np.nan, np.nan
    
    total_return = (1 + returns_clean).prod() - 1
    n_periods = len(returns_clean)
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    annual_volatility = returns_clean.std() * np.sqrt(periods_per_year)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    
    return annual_return, annual_volatility, sharpe_ratio

# 计算各项指标
ls_returns = portfolio_results['long_short_return']
ls_annual_return, ls_annual_vol, ls_sharpe = calculate_performance_metrics(ls_returns)

q1_returns = portfolio_results['Q1_return']
q5_returns = portfolio_results['Q5_return']
q1_annual_return, q1_annual_vol, q1_sharpe = calculate_performance_metrics(q1_returns)
q5_annual_return, q5_annual_vol, q5_sharpe = calculate_performance_metrics(q5_returns)

# ==================== 步骤6：结果显示 ====================
print("\n" + "="*80)
print("MAX因子多空组合绩效结果（基于日度数据）")
print("="*80)
print(f"{'组合':<25} {'年化收益率':<12} {'年化波动率':<12} {'夏普比率':<10} {'月数':<6}")
print("-"*80)
print(f"{'Q1 (低MAX)':<25} {q1_annual_return:>10.2%} {q1_annual_vol:>11.2%} {q1_sharpe:>9.2f} {len(q1_returns):>6}")
print(f"{'Q5 (高MAX)':<25} {q5_annual_return:>10.2%} {q5_annual_vol:>11.2%} {q5_sharpe:>9.2f} {len(q5_returns):>6}")
print(f"{'多空组合 (Q5-Q1)':<25} {ls_annual_return:>10.2%} {ls_annual_vol:>11.2%} {ls_sharpe:>9.2f} {len(ls_returns):>6}")

# 计算累计收益
portfolio_results['Q1_cumulative'] = (1 + portfolio_results['Q1_return']).cumprod()
portfolio_results['Q5_cumulative'] = (1 + portfolio_results['Q5_return']).cumprod()
portfolio_results['LS_cumulative'] = (1 + portfolio_results['long_short_return']).cumprod()

# 详细统计
win_rate = (portfolio_results['long_short_return'] > 0).mean()
avg_monthly_return = portfolio_results['long_short_return'].mean()
max_monthly_return = portfolio_results['long_short_return'].max()
min_monthly_return = portfolio_results['long_short_return'].min()

print(f"\n详细统计信息：")
print(f"多空组合平均月度收益率：{avg_monthly_return:.2%}")
print(f"多空组合胜率：{win_rate:.2%}")
print(f"最大单月收益：{max_monthly_return:.2%}")
print(f"最大单月亏损：{min_monthly_return:.2%}")
print(f"回测期间：{portfolio_results['year_month'].min()} 到 {portfolio_results['year_month'].max()}")

# ==================== 步骤7：数据保存 ====================
# 保存处理后的数据
df_processed.to_csv('processed_data_with_max.csv', index=False, encoding='utf-8-sig')
portfolio_results.to_csv('portfolio_results.csv', index=False, encoding='utf-8-sig')
print(f"\n数据已保存：")
print("- processed_data_with_max.csv：包含MAX因子的完整数据")
print("- portfolio_results.csv：组合绩效结果")

# ==================== 步骤8：可视化 ====================
plt.figure(figsize=(12, 10))

# 子图1：累计收益曲线
plt.subplot(2, 2, 1)
months_str = portfolio_results['year_month'].astype(str)

plt.plot(months_str, portfolio_results['Q1_cumulative'], 
         label='Q1组合 (低MAX)', linewidth=2, alpha=0.7)
plt.plot(months_str, portfolio_results['Q5_cumulative'], 
         label='Q5组合 (高MAX)', linewidth=2, alpha=0.7)
plt.plot(months_str, portfolio_results['LS_cumulative'], 
         label='多空组合 (Q5-Q1)', linewidth=3)

plt.title('MAX因子多空组合累计收益曲线', fontsize=12, fontweight='bold')
plt.xlabel('月份')
plt.ylabel('累计净值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 子图2：月度收益率分布
plt.subplot(2, 2, 2)
plt.hist(portfolio_results['long_short_return'], bins=20, alpha=0.7, color='steelblue')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
plt.title('多空组合月度收益率分布', fontsize=12, fontweight='bold')
plt.xlabel('月度收益率')
plt.ylabel('频数')
plt.grid(True, alpha=0.3)

# 子图3：月度收益率时间序列
plt.subplot(2, 2, 3)
plt.bar(range(len(portfolio_results)), portfolio_results['long_short_return'], 
        alpha=0.7, color=['green' if x > 0 else 'red' for x in portfolio_results['long_short_return']])
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('多空组合月度收益率', fontsize=12, fontweight='bold')
plt.xlabel('月份序号')
plt.ylabel('月度收益率')
plt.grid(True, alpha=0.3)

# 子图4：Q1 vs Q5收益率散点图
plt.subplot(2, 2, 4)
plt.scatter(portfolio_results['Q1_return'], portfolio_results['Q5_return'], 
           alpha=0.6, c=portfolio_results['long_short_return'], cmap='coolwarm')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.plot([-0.2, 0.2], [-0.2, 0.2], 'k--', alpha=0.3)  # 45度线
plt.title('Q1 vs Q5月度收益率', fontsize=12, fontweight='bold')
plt.xlabel('Q1收益率 (低MAX)')
plt.ylabel('Q5收益率 (高MAX)')
plt.colorbar(label='多空组合收益')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('max_factor_portfolio_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 步骤9：显示前12个月详细结果 ====================
print("\n" + "="*85)
print("前12个月详细表现：")
print("="*85)
print(f"{'月份':<12} {'Q1收益':<10} {'Q5收益':<10} {'多空收益':<12} {'股票数':<8} {'交易日':<8}")
print("-"*85)
for i, row in portfolio_results.head(12).iterrows():
    print(f"{str(row['year_month']):<12} {row['Q1_return']:>8.2%} {row['Q5_return']:>8.2%} {row['long_short_return']:>10.2%} {row['n_stocks']:>8} {row['n_trading_days']:>8}")

print("\n" + "="*80)
print("分析完成！")
print("="*80)