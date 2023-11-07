import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from datetime import timedelta

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
from scipy import stats
from IPython.display import display
from collections import Counter

def prepare_backtesting_data(df_daily_return, df_actions, benchmark, folder):

    start_date = str(df_daily_return.reset_index()['date'].iloc[0])
    end_date = str(df_daily_return.reset_index()['date'].iloc[-1])
    
    crypto_list = df_actions.columns.tolist()

    ohlcv = pd.read_csv(f'{folder}ohlcv.csv', index_col=0, parse_dates=True)
    ohlcvs = []
    for i in range(len(crypto_list)):
        temp = ohlcv.loc[ohlcv['ticker']==crypto_list[i]].resample('8h').ffill().loc[start_date:end_date].reset_index()
        ohlcvs.append(temp)
    ohlcv = pd.concat([ohlcvs[i] for i in range(len(ohlcvs))], axis=0).set_index(['date', 'ticker']).sort_index().reset_index()
    
    close_data = pd.crosstab(index=ohlcv['date'], columns=ohlcv['ticker'], values=ohlcv['close'], aggfunc=lambda s:s)
    close_data.index = pd.to_datetime(close_data.index, format="%Y-%m-%d %H:%M:%S")
    
    # bm
    ohlcv = pd.read_csv(f'{folder}ohlcv.csv', index_col=0, parse_dates=True)
    for i in range(len(crypto_list)):
        bm = ohlcv.loc[ohlcv['ticker']==benchmark].resample('8h').ffill().loc[start_date:end_date].reset_index()
    
    bm = pd.crosstab(index=bm['date'], columns=bm['ticker'], values=bm['close'], aggfunc=lambda s:s)
    bm.index = pd.to_datetime(bm.index, format="%Y-%m-%d %H:%M:%S")
    
    return bm, close_data

def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

def highlight_max(data, color='tomato'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

def highlight_min(data, color='black'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # from .apply(axis=None)
        is_min = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''),
                            index=data.index, columns=data.columns)
        
def cumulative_returns(prices):
    return (1 + prices.pct_change(1)).cumprod()

def drawdown(prices):
    rets = cumulative_returns(prices)
    return (rets.div(rets.cummax()) - 1) * 100

def calc_max_drawdown(prices):
    return (prices / prices.expanding(min_periods=1).max()).min() - 1

def calc_cagr(prices):
    start = prices.index[0]
    end = prices.index[-1]
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / year_frac(start, end)) - 1

def year_frac(start, end):
    return (end - start).total_seconds() / (31557600)

def to_drawdown_series(prices):
    drawdown = prices.copy()
    drawdown = drawdown.fillna(method='ffill')
    drawdown[np.isnan(drawdown)] = -np.Inf
    roll_max = np.maximum.accumulate(drawdown)
    drawdown = drawdown / roll_max - 1.
    return drawdown

def alpha(returns, bm_returns, _beta=None):
    
    if len(returns) < 2:
        return np.nan

    if _beta is None:
        b = beta(returns, bm_returns)
    else:
        b = _beta
    
    alpha_series = returns - (b * bm_returns)

    return alpha_series.mean() * 365

def beta(returns, bm_returns):
    
    if len(returns) < 2 or len(bm_returns) < 2:
        return np.nan
    
    # Filter out dates with np.nan as a return value
    joint = pd.concat([returns, bm_returns], axis=1).dropna()
    
    if len(joint) < 2:
        return np.nan
    
    if np.absolute(joint.var().iloc[1]) < 1.0e-30:
        return np.nan

    return np.cov(joint.values.T, ddof=0)[0, 1] / np.var(joint.iloc[:, 1])

def alpha_beta(returns, bm_returns):
    
    b = beta(returns, bm_returns)
    
    a = alpha(returns, bm_returns, _beta=b)
    
    return a, b

def omega_ratio(returns):

    if len(returns) < 2:
        return np.nan

    numer = sum(returns[returns > 0.0])
    denom = -1.0 * sum(returns[returns < 0.0])

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan

def information_ratio(returns, bm_returns):
    
    if len(returns) < 2:
        return np.nan

    active_return = returns - bm_returns
    tracking_error = np.std(active_return, ddof=1)
    
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    
    return np.mean(active_return) * 365 ** (0.5) / tracking_error

def tail_ratio(returns):
    
    returns = returns.dropna()
    
    if len(returns) < 1:
        return np.nan

    return np.abs(np.percentile(returns, 95)) / np.abs(np.percentile(returns, 5))


def stability_of_timeseries(returns):
    
    if len(returns) < 2:
        return np.nan

    returns = returns.dropna()

    cum_log_returns = np.log1p(returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)), cum_log_returns.values)[2]

    return rhat ** 2

def backtest_analytics(df_daily_return, bm, df_actions, close_data, cash, NAME='test', BM_NAME='benchmark'):

    # quantity over time preparation
    fees = 0
    last_alloc = {}
    allocation_list, trans_allocation_list = [], []
    
    for i in range(len(df_actions)):

        temp_actions = df_actions.iloc[i].loc[df_actions.iloc[i]>=0].to_frame().reset_index()
        temp_actions.columns = ['ticker', 'weight']

        temp_tickers = temp_actions['ticker'].tolist()

        temp_actions['allocate_value'] = temp_actions['weight'] * cash
        temp_actions['price'] = close_data[temp_tickers].iloc[i].values
        temp_actions['quantity'] = temp_actions['allocate_value']/temp_actions['price']
        temp_actions['quantity'] = temp_actions[['ticker','quantity']].apply(lambda x: round(x['quantity'],4) if x['ticker'] in ['ETH/USDT', 'BTC/USDT', 'YFI/USDT'] else round(x['quantity'],1), axis=1)
        temp_actions['quantity'] = temp_actions[['ticker','quantity']].apply(lambda x: int(x['quantity']) if x['ticker'] in ['USDC/USDT'] else x['quantity'], axis=1)

        alloc = dict(temp_actions[['ticker', 'quantity']].values)

        last_counter = Counter(last_alloc)
        now_counter = Counter(alloc)
        now_counter.subtract(last_counter)
        now_counter = dict(now_counter)

        for k, v in now_counter.items():
            if k in ['ETH/USDT', 'BTC/USDT', 'YFI/USDT']:
                now_counter[k] = round(v, 4)
            elif k in ['USDC/USDT']:
                now_counter[k] = int(v)
            else:
                now_counter[k] = round(v, 1)
                
        final_alloc_df = pd.DataFrame.from_dict(now_counter, orient='index')
        final_alloc_df.columns = ['quantity']
        
        temp_actions['quantity'] = final_alloc_df['quantity'].values.tolist()
        temp_actions['quantity'] = temp_actions[['price','quantity']].apply(lambda x: 0 if abs(x['quantity']*x['price']) <= 10 else x['quantity'], axis=1)
        
        temp_actions['transaction'] = temp_actions['quantity'] * temp_actions['price'] / 500
        fees += temp_actions['transaction'].sum()
        
        now_counter = dict(temp_actions[['ticker', 'quantity']].values)

        now = Counter(now_counter)
        original = Counter(last_alloc)

        last_alloc = dict(now + original)
        cash = df_daily_return['account_value'].iloc[i]
        
        allocation_df = pd.Series(last_alloc, name='quantity').to_frame().sort_index().T
        allocation_list.append(allocation_df)
        
        trans_allocation_df = pd.Series(now_counter, name='quantity').to_frame().sort_index().T
        trans_allocation_list.append(trans_allocation_df)
        
    df_shares = pd.concat(allocation_list).fillna(0)
    df_shares.index = close_data.index
    
    df_trans_share = pd.concat(trans_allocation_list).fillna(0)
    df_trans_share.index = close_data.index
    
    df_daily_return['daily_return'] = df_daily_return['account_value'].pct_change().fillna(0)

    # roi preparation
    initial_capital = df_daily_return['account_value'].iloc[0]
    df_daily_return['Cumulative ROI'] = df_daily_return['account_value'].apply(lambda x: ((x - initial_capital) / initial_capital)*100)

    bm = (bm[BM_NAME] * initial_capital / bm[BM_NAME].iloc[0]).reset_index()
    bm.columns = ['date', 'account_value']

    START_DATE = str(bm['date'].iloc[0])
    END_DATE = str(bm['date'].iloc[-1])

    bm['daily_return'] = bm['account_value'].pct_change().fillna(0)
    bm['Cumulative ROI'] = bm['account_value'].apply(lambda x: ((x - initial_capital) / initial_capital)*100)
    bm['date'] = pd.to_datetime(bm['date'], format='%Y-%m-%d %H:%M:%S')
    bm = bm.set_index('date')
    
    # daily drawdown preparation
    df_daily_dd = to_drawdown_series(df_daily_return['account_value']).reset_index()
    df_daily_dd.columns = ['date', 'Daily Drawdown']
    df_daily_dd['Max Daily Rolling Drawdown'] = df_daily_dd['Daily Drawdown'].rolling(window=365, min_periods=1).min()

    # metrics of trading performance preparation
    sr = df_daily_return['daily_return'].mean()/df_daily_return['daily_return'].std()*365**(0.5)
    bm_sr = bm['daily_return'].mean()/bm['daily_return'].std()*365**(0.5)
    sr_str = str(round(sr,2))
    bm_sr_str = str(round(bm_sr,2))

    cr = calc_cagr(df_daily_return['account_value']) / abs(calc_max_drawdown(df_daily_return['account_value']))
    bm_cr = calc_cagr(bm['account_value']) / abs(calc_max_drawdown(bm['account_value']))
    cr_str = str(round(cr,2))
    bm_cr_str = str(round(bm_cr,2))

    cagr = calc_cagr(df_daily_return['account_value'])
    bm_cagr = calc_cagr(bm['account_value'])
    cagr_str = str(round(cagr*100,2))+'%'
    bm_cagr_str = str(round(bm_cagr*100,2))+'%'

    meandd = drawdown(df_daily_return['account_value']).mean()
    bm_meandd = drawdown(bm['account_value']).mean()
    meandd_str = str(round(meandd,2))+'%'
    bm_meandd_str = str(round(bm_meandd,2))+'%'

    maxdd = calc_max_drawdown(df_daily_return['account_value'])
    bm_maxdd = calc_max_drawdown(bm['account_value'])
    maxdd_str = str(round(maxdd*100,2))+'%'
    bm_maxdd_str = str(round(bm_maxdd*100,2))+'%'

    prob_lose_money = (df_daily_return['Cumulative ROI'][df_daily_return['Cumulative ROI']<0].shape[0] / df_daily_return['Cumulative ROI'].shape[0])
    bm_prob_lose_money = (bm['Cumulative ROI'][bm['Cumulative ROI']<0].shape[0] / bm['Cumulative ROI'].shape[0])
    prob_lose_money_str = str(round(prob_lose_money*100,2))+'%'
    bm_prob_lose_money_str = str(round(bm_prob_lose_money*100,2))+'%'
    
    a, b = alpha_beta(df_daily_return['daily_return'], bm['daily_return'])
    a_str = str(round(a*100,2))+'%'
    b_str = str(round(b,2))
    
    i = information_ratio(df_daily_return['daily_return'], bm['daily_return'])
    i_str = str(round(i,2))

    t = tail_ratio(df_daily_return['daily_return'])
    bm_t = tail_ratio(bm['daily_return'])
    t_str = str(round(t,2))
    tt_str = str(round(1/t,2))
    bm_t_str = str(round(bm_t,2))

    s = stability_of_timeseries(df_daily_return['daily_return'])
    bm_s = stability_of_timeseries(bm['daily_return'])
    s_str = str(round(s,2))
    bm_s_str = str(round(bm_s,2))
    
    o = omega_ratio(df_daily_return['daily_return'])
    bm_o = omega_ratio(bm['daily_return'])
    o_str = str(round(o,2))
    bm_o_str = str(round(bm_o,2))

    # monthly & yearly return preparation
    monthly_return = pd.concat([df_daily_return['account_value'][:1], df_daily_return['account_value'].resample('M').ffill()]).pct_change()[1:]
    monthly_return = monthly_return.apply(lambda x:round(x, 4))
    monthly_return = monthly_return.reset_index()
    monthly_return['month'] = monthly_return['date'].apply(lambda x:str(x)[5:7])
    monthly_return['year'] = monthly_return['date'].apply(lambda x:str(x)[:4])
    monthly_return_df = pd.crosstab(index=monthly_return['year'], columns=monthly_return['month'], values=monthly_return['account_value'], aggfunc=lambda s:s)

    yearly_return = pd.concat([df_daily_return['account_value'][:1], df_daily_return['account_value'].resample('Y').ffill()]).pct_change()[1:]
    yearly_return = yearly_return.apply(lambda x:round(x, 4))
    yearly_return = yearly_return.reset_index()
    monthly_return_df['total'] = yearly_return['account_value'].values

    return_df = monthly_return_df.style.applymap(color_negative_red).apply(highlight_max, axis=None).apply(highlight_min, axis=None).format("{:.2%}")

    bm_monthly_return = pd.concat([bm['account_value'][:1], bm['account_value'].resample('M').ffill()]).pct_change()[1:]
    bm_monthly_return = bm_monthly_return.apply(lambda x:round(x, 4))
    bm_monthly_return = bm_monthly_return.reset_index()
    bm_monthly_return['month'] = bm_monthly_return['date'].apply(lambda x:str(x)[5:7])
    bm_monthly_return['year'] = bm_monthly_return['date'].apply(lambda x:str(x)[:4])
    bm_monthly_return_df = pd.crosstab(index=bm_monthly_return['year'], columns=bm_monthly_return['month'], values=bm_monthly_return['account_value'], aggfunc=lambda s:s)

    bm_yearly_return = pd.concat([bm['account_value'][:1], bm['account_value'].resample('Y').ffill()]).pct_change()[1:]
    bm_yearly_return = bm_yearly_return.apply(lambda x:round(x, 4))
    bm_yearly_return = bm_yearly_return.reset_index()
    bm_monthly_return_df['total'] = bm_yearly_return['account_value'].values

    bm_return_df = bm_monthly_return_df.style.applymap(color_negative_red).apply(highlight_max, axis=None).apply(highlight_min, axis=None).format("{:.2%}")
    
    # start plotting
    df_daily_return = df_daily_return.reset_index()
    bm = bm.reset_index()

    # Trading Performance (Portfolio vs Benchmark)
    metrics = pd.Series([cagr, sr, meandd/100, prob_lose_money, maxdd, cr])
    metrics = metrics.apply(lambda x:round(x,2))
    bm_metrics = pd.Series([bm_cagr, bm_sr, bm_meandd/100, bm_prob_lose_money, bm_maxdd, bm_cr])
    bm_metrics = bm_metrics.apply(lambda x:round(x,2))

    metrics_list = [f'CAGR : {cagr_str}', f'Sharpe Ratio : {sr_str}', f'Average Drawdown : {meandd_str}',
                    f'Probability of Losing Money : {prob_lose_money_str}', f'Maximum Drawdown : {maxdd_str}', f'Calmar Ratio : {cr_str}']
    names = [NAME, BM_NAME]

    metrics_df = pd.DataFrame()
    metrics_df[names[0]] = metrics
    metrics_df[names[1]] = bm_metrics
    metrics_df = metrics_df.T.reset_index(drop=True)

    fig = go.Figure()

    for i in range(len(metrics_df)):
        fig.add_trace(go.Scatterpolar(r = metrics_df.loc[i].values, theta = metrics_list, fill='toself', name = names[i], showlegend=True))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title=f"Trading Performance (Portfolio vs {BM_NAME})", template="plotly_dark")
    fig.update_traces(fill='toself', opacity=0.4)
    fig.show()

    # Portfolio Growth
    fig = px.line(df_daily_return, x='date', y='account_value', title = 'Portfolio Growth')
    fig.update_layout(hovermode = "y", yaxis_tickformat = "000")
    fig.show()

    # Cumulative Performance (Portfolio vs Benchmark)
    fig = go.Figure()
    fig.add_scatter(x = df_daily_return['date'], y = df_daily_return['Cumulative ROI'], mode = 'lines', name = NAME)
    fig.add_scatter(x = bm['date'], y = bm['Cumulative ROI'], mode = 'lines', name = BM_NAME)
    fig.add_shape(type = 'line', x0 = START_DATE, y0 = 0, x1 = END_DATE, y1 = 0, line = dict(color = 'black'))

    fig.update_layout(hovermode = "x", yaxis = dict(ticksuffix = "%"), title = f'Cumulative Performance (Portfolio vs {BM_NAME})')
    fig.show()
    
    # Drawdown Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df_daily_dd['date'], y = df_daily_dd['Daily Drawdown'], 
                                fill = 'tozeroy', name = 'Daily Drawdown'))
    fig.add_trace(go.Scatter(x = df_daily_dd["date"], y = df_daily_dd['Max Daily Rolling Drawdown'], 
                                mode = 'lines', name = 'Max Daily Drawdown'))
    fig.update_layout(hovermode = "x", yaxis_tickformat = "%", title = 'Portfolio Drawdown')
    fig.show()

    # Multi-Distribution of Daily Returns (Portfolio vs Benchmark)
    hist_data = [np.array(df_daily_return['account_value'].pct_change()[1:]), np.array(bm['account_value'].pct_change()[1:])]
    group_labels = [NAME, BM_NAME]
    colors = ['#636EFA', '#EF553B']
    fig = ff.create_distplot(hist_data, group_labels, bin_size = 0.001, colors = colors)
    fig.update_layout(yaxis = dict(ticksuffix = "%"), xaxis_tickformat = "%", title = f'Distribution of Daily Return (Portfolio vs {BM_NAME})')
    fig.show()
    
    # Change of Shares Over Time
    shares_df = df_shares.unstack().reset_index()
    shares_df.columns = ['ticker', 'date', 'share']
    shares_df = shares_df.sort_values(['date', 'ticker']).reset_index(drop=True)
    fig = px.line(shares_df, x="date", y="share", color="ticker", title = 'Shares Over Time')
    fig.show()

    print("\n===== Daily Shares =====\n")
    display(df_shares)

    print("\n===== Daily Change of Shares =====\n")
    display(df_trans_share)
    
    # calculate slippage and fee : 2/1000 
    print("\n===== Transaction Fees (0.2%) =====\n")
    print(fees)

    print("\n===== Monthly return =====\n")
    display(return_df)

    print("\n===== Benchmark Monthly return =====\n")
    display(bm_return_df)
    
    performance = pd.DataFrame()
    performance['Performance Metrics'] = ['Alpha','Beta','Information Ratio','CAGR','Sharp Ratio',
                                          'Calmar Ratio','Omega Ratio','Mean Drawdown','Max Drawdown',
                                          'Prob of Losing Money','Stability','Tail Ratio *']
    performance[NAME] = [a_str,b_str,i_str,cagr_str,sr_str,cr_str,o_str,
                         meandd_str,maxdd_str,prob_lose_money_str,s_str,t_str]
    performance[BM_NAME] = ['*','*','*',bm_cagr_str,bm_sr_str,bm_cr_str,bm_o_str,
                            bm_meandd_str,bm_maxdd_str,bm_prob_lose_money_str,bm_s_str,bm_t_str]
    performance = performance.set_index('Performance Metrics')
    
    print('\n===== Performance Report =====\n')
    display(performance)
    
    print(f"\n* means losses are {tt_str} as bad as profits")
    