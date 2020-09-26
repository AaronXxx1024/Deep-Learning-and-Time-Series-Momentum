#%% Fin 794 - Group Project: Enhancing Time Series Momentum Strategies Using Deep Neural Networks
""" Created for Backtesting specific TSMOM set up in paper - Han (Aaron) Xiao"""
# TSMOM Part
# Part 1 - Data Preparation
# Load module
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pytz
import empyrical

#%% Read data
# ast means Asset
future_url = "/Users/aaronx-mac/PycharmProjects/Learning/Fin794 Group Project/futures.csv"
ast = pd.read_csv(future_url)

# 发现"Date"列的type是object，将其转为时间
ast.info()
ast['Date'] = pd.to_datetime(ast['Date'], format='%Y-%m-%d')
ast.set_index('Date', inplace=True)
ast.info()

#%% Part 2 - Asset Overview
# 获得monthly column
std_index = ast.resample('BM').last().index
mth_index = pd.DataFrame(index=std_index)
mth_index_vol = pd.DataFrame(index=std_index)
summary_stats = pd.DataFrame(index=['Asset', 'Start', 'Mean', 'Std', 'Skew', 'Kurt', 'Sharpe Ratio'])

#%%
for i in ast.columns:
    returns = ast[i]
    returns.dropna(inplace=True)
    # store this to show when data series starts
    first_date = returns.index[0].strftime("%Y-%m-%d")

    # 计算每个future的累计回报
    ret_index = (1 + returns).cumprod()
    ret_index[0] = 1

    # scale asset returns with an ex-ante volatility estimate
    # computed using an exponentially weighted moving standard deviation with a 60-day span
    # which is "Sigma" in paper
    day_vol = returns.ewm(ignore_na=False,
                          adjust=True,
                          com=60,
                          min_periods=0).std(bias=False)
    # annualized
    vol = day_vol * np.sqrt(261)

    ret_index = pd.concat([ret_index, vol], axis=1)
    ret_index.columns = [i, 'vol']

    # convert to monthly return
    ret_m_index = ret_index.resample('BM').last().ffill()
    ret_m_index.ix[0][i] = 1
    # 月累计回报
    mth_index = pd.concat([mth_index, ret_m_index[i]], axis=1)
    # 每个asset的年vol（不是60天那个）
    tmp = ret_m_index['vol']
    tmp.name = i + "_Vol"
    mth_index_vol = pd.concat([mth_index_vol, tmp], axis=1)
    # 年均回报
    tmp_mean = ret_index[i].pct_change().mean() * 252
    # 年std
    tmp_std = ret_index[i].pct_change().std() * np.sqrt(252)
    # 年偏度
    tmp_skew = ret_index[i].pct_change().skew()
    # 年峰度
    tmp_kurt = ret_index[i].pct_change().kurt()
    # 年均sharpe ratio
    sr = tmp_mean / tmp_std

    dict = {'Asset': i,
            'Start': first_date,
            'Mean': np.round(tmp_mean, 4),
            'Std': np.round(tmp_std, 4),
            'Skew': np.round(tmp_skew, 4),
            'Kurt': np.round(tmp_kurt, 4),
            'Sharpe Ratio': np.round(sr, 4),
            }
    summary_stats[i] = pd.Series(dict)

# Sigma_tgt target in paper
vol_tgt = 0.15
#%% test sharpe of one of the asset i
# print(empyrical.sharpe_ratio(ast['AD1'], period='daily'))

#%%
summary_stats = summary_stats.transpose()
futures_list_url = "/Users/aaronx-mac/PycharmProjects/Learning/Fin794 Group Project/futures_list.csv"
futures_list = pd.read_csv(futures_list_url)
all = summary_stats.reset_index().merge(futures_list)
all.sort_values(by=["ASSET_CLASS", "FUTURES"], inplace=True)
del all['Asset'], all['index']

#%% Annualized Performance for all future contracts
all.set_index(['ASSET_CLASS', 'FUTURES']).style.set_properties(**{'text-align': 'right'})

#%% Part 3 - Trading Strategy: TSMOM with Volatility Scaling (1980 - 2019)
# Trend Estimation & Position Sizing
S_k = [8, 16, 32]
L_k = [24, 48, 96]
del returns
del tmp
daily_index = ast.index
Y_bar = pd.DataFrame(index=daily_index)
X_t = pd.DataFrame(index=daily_index)
cum_return = pd.DataFrame(index=daily_index)
Signal = pd.DataFrame(index=daily_index)
for i in ast.columns:
    returns = ast[i]
    returns.dropna(inplace=True)

    # 计算每个future的累计回报
    ast_cum_return = (1 + returns).cumprod()
    ast_cum_return[0] = 1

    # Create Final Trend Estimation: Y_bar [Equation(8) in paper]
    for p in range(0,len(S_k)):
        m_i_S = ast_cum_return.ewm(ignore_na=False,
                                    adjust=True,
                                    halflife= np.log(0.5)/np.log(1 - 1/S_k[p]),
                                    min_periods=0).mean()

        m_i_L = ast_cum_return.ewm(ignore_na=False,
                                    adjust=True,
                                    halflife=np.log(0.5) / np.log(1 - 1 / L_k[p]),
                                    min_periods=0).mean()

        MACD = m_i_S - m_i_L
        # We normalize with a moving standard deviation as a measure of the realized 3-months normal volatility (PW=63)
        q_t = MACD / ast_cum_return.rolling(63).std()
        # We normalize this series with its realized standard deviation over the short window (SW=252)
        Y_t = q_t/q_t.rolling(252).std()
        Y_bar = pd.concat([Y_bar, Y_t], axis=1)

    Y_bar[i + '_Mean'] = Y_bar.mean(1)
    # Create Final Position Sizing: X_t [Equation(7) in paper]
    # X_t should be series
    X_t[i + '-X_t'] = Y_bar[i + '_Mean'] * np.exp((-1) * np.square(Y_bar[i + '_Mean']) / 4) / 0.89

    tmp = Y_bar[i + '_Mean']
    tmp_1 = X_t[i + '-X_t']
    cum_return = pd.concat([cum_return, ast_cum_return], axis=1)
    Signal = pd.concat([Signal, ast_cum_return, tmp, tmp_1], axis=1)

#%% Background Setting
ind_return = pd.DataFrame(index=daily_index)
strategy_cumm_rtns = pd.DataFrame(index=daily_index)
df = pd.DataFrame(index=daily_index)

#%%
tolerance = 0.
look_back = 1 # Note: daily update, based on turnover analysis in the paper, I believe author used daily here

# Vol scaling
vol_flag = 1                  # Set flag to 1 for vol targeting
if vol_flag == 1:
    target_vol = 0.15
else:
    target_vol = 'no target vol'

#%% Part 4 - Trading
# Equation(1) in paper, daily basis
for i in ast:
    # 进行水平方向的合并
    df = pd.concat([ast[i], X_t[i + "-X_t"]], axis=1)

    day_vol = df[i].ewm(ignore_na=False,
                          adjust=True,
                          span=60,
                          min_periods=0).std(bias=False)

    # daily return based on equation (1) for individual asset
    df[i + '-ind_return'] = df[i] * df[i + "-X_t"] * target_vol / day_vol

    # convert to daily return for different asset in t
    ind_return = pd.concat([ind_return, df[i + '-ind_return']], axis=1)

#%% Portfolio Average
# Strategy return in daily
ind_return['port_avg'] = ind_return.mean(skipna=1, axis=1)
# convert to monthly basis
strategy_month_rtns = ind_return['port_avg'].resample('BM').last().ffill()

strategy_cumm_rtns['cummulative'] = (1 + ind_return['port_avg']).cumprod()
# convert to monthly cum return
strategy_month = strategy_cumm_rtns['cummulative'].resample('BM').last().ffill()

#%% Print Results
print("Annualized Sharpe Ratio = ", empyrical.sharpe_ratio(ind_return['port_avg'], period='daily'))
print("Annualized Mean Returns = ", empyrical.annual_return(ind_return['port_avg'], period='daily'))
print("Annualized Standard Deviations = ", empyrical.annual_volatility(ind_return['port_avg'], period='daily'))
print("Max Drawdown (MDD) = ", empyrical.max_drawdown(ind_return['port_avg']))
print("Sortino ratio = ", empyrical.sortino_ratio(ind_return['port_avg'], period='daily'))
print("Calmar ratio = ", empyrical.calmar_ratio(ind_return['port_avg'], period='daily'))

#%% Visualization
#print(empyrical.sharpe_ratio(strategy_month_rtns, period='monthly'))
a = empyrical.cum_returns(ind_return['port_avg'])
#b = strategy_month
plt.plot(a, color = 'red', label = 'Raw Portfolio')
plt.title('Cumulative return in daily basis')
plt.xlabel('Time')
plt.ylabel('Cumulative return')
plt.legend()
plt.show()