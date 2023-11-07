import warnings
warnings.filterwarnings('ignore')

from joblib import delayed, Parallel, Memory, cpu_count
from pycatch22 import catch22_all
from talib import WMA
import pandas as pd
import numpy as np
import joblib
import talib
import ta

def tr(data):
    data['previous_close'] = data['close'].shift(1)
    data['high-low'] = abs(data['high'] - data['low'])
    data['high-pc'] = abs(data['high'] - data['previous_close'])
    data['low-pc'] = abs(data['low'] - data['previous_close'])

    tr = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)

    return tr

def atr(data, period):
    data['tr'] = tr(data)
    atr = data['tr'].rolling(period).mean()

    return atr

def supertrend(df, period=7, atr_multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    df['atr'] = atr(df, period)
    df['upperband'] = hl2 + (atr_multiplier * df['atr'])
    df['lowerband'] = hl2 - (atr_multiplier * df['atr'])
    df['in_uptrend'] = True

    for current in range(1, len(df.index)):
        previous = current - 1

        if df['close'][current] > df['upperband'][previous]:
            df['in_uptrend'][current] = True
        elif df['close'][current] < df['lowerband'][previous]:
            df['in_uptrend'][current] = False
        else:
            df['in_uptrend'][current] = df['in_uptrend'][previous]

            if df['in_uptrend'][current] and df['lowerband'][current] < df['lowerband'][previous]:
                df['lowerband'][current] = df['lowerband'][previous]

            if not df['in_uptrend'][current] and df['upperband'][current] > df['upperband'][previous]:
                df['upperband'][current] = df['upperband'][previous]

    df['in_uptrend'] = df['in_uptrend'].apply(lambda x:int(x))
        
    return df

def returns(prices):
    """
    Calulates the growth of 1 dollar invested in a stock with given prices
    """
    return (1 + prices.pct_change(1)).cumprod()

def drawdown(prices):
    """
    Calulates the drawdown of a stock with given prices
    """
    rets = returns(prices)
    return (rets.div(rets.cummax()) - 1) * 100

def prepare_data(folder, crypto_list, feature_list, train_start_date, train_end_date, trade_start_date, trade_end_date):

    ohlcv = pd.read_csv(f'{folder}ohlcv.csv', index_col=0, parse_dates=True)
    ohlcvs = []
    for i in range(len(crypto_list)):
        temp = ohlcv.loc[ohlcv['ticker']==crypto_list[i]].resample('8h').ffill().loc[train_start_date:].reset_index()
        ohlcvs.append(temp)
    ohlcv = pd.concat([ohlcvs[i] for i in range(len(ohlcvs))], axis=0).set_index(['date', 'ticker']).sort_index().reset_index()

    close_data = pd.crosstab(index=ohlcv['date'], columns=ohlcv['ticker'], values=ohlcv['close'], aggfunc=lambda s:s)
    #first_date_in_market_df = close_data.apply(pd.Series.first_valid_index).to_frame().reset_index()
    #first_date_in_market_df.columns = ['ticker', 'first_date_in_market']
    close_data = close_data.dropna(axis=1)
    close_data.index = pd.to_datetime(close_data.index, format="%Y-%m-%d %H:%M:%S")

    ticker_list = close_data.columns.tolist()

    if 'hedging' in feature_list:
        
        FREQ = 'W'
        MOMENTUM = 'BTC/USDT'
        momentum = MOMENTUM.lower()

        mom_close = close_data[[MOMENTUM]]

        returns_1w = mom_close[MOMENTUM].resample(FREQ).ffill().pct_change()
        returns_3w = mom_close[MOMENTUM].resample(FREQ).ffill().pct_change().add(1).rolling(3).apply(np.prod)
        returns_6w = mom_close[MOMENTUM].resample(FREQ).ffill().pct_change().add(1).rolling(6).apply(np.prod)

        returns_mom = ((returns_1w + (returns_3w-1) + (returns_6w-1))/3).to_frame().reset_index()
        returns_mom.columns = ['date', 'hedging']
        returns_mom['hedging'] = returns_mom['hedging'].fillna(method='bfill')
        returns_mom = returns_mom.set_index('date')
        returns_mom = returns_mom.reindex(mom_close.index, method='ffill').fillna(method='bfill').reset_index()
        returns_mom.columns = ['date', 'hedging']
        returns_mom['date'] = pd.to_datetime(returns_mom['date'], format='%Y-%m-%d %H:%M:%S')
    
    if 'dd' in feature_list:

        dd_df = close_data.copy()

        for i in range(len(ticker_list)):
            dd_df[ticker_list[i]] = drawdown(close_data[ticker_list[i]])

        dd_df = dd_df.fillna(0)
        dd_data = dd_df.unstack().reset_index()
        dd_data.columns = ['ticker', 'date', 'dd']
    
    if '86ta' in feature_list:

        success = []

        for i in range(len(ticker_list)):
            temp = ohlcv.loc[ohlcv['ticker']==ticker_list[i]].reset_index(drop=True).drop(['ticker'], axis=1)
            ta_df = ta.add_all_ta_features(temp, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
            ta_df['ticker'] = ticker_list[i]
            success.append(ta_df)

        ohlcv_ta = pd.concat([success[i] for i in range(len(success))], axis=0).set_index(['date', 'ticker']).sort_index()

    if 'st' in feature_list:

        success = []

        for i in range(len(ticker_list)):
            temp = ohlcv.loc[ohlcv['ticker']==ticker_list[i]].reset_index().drop(['ticker'], axis=1)
            st_df = supertrend(temp).fillna(method='ffill').fillna(method='bfill')
            st_df['Supertrend'] = st_df['in_uptrend'].diff().fillna(0).values
            st_df['ticker'] = ticker_list[i]
            
            success.append(st_df[['date', 'ticker', 'upperband', 'lowerband', 'Supertrend']])

        ohlcv_st = pd.concat([success[i] for i in range(len(success))], axis=0).set_index(['date', 'ticker']).sort_index()

    if 'qlib' in feature_list:

        windows=[2, 3, 4, 5]
        ratio_range=range(2, 5)

        def get_qlib(data):

            data_columns = data.columns

            for lb in windows:
                data[f'open_close_ratio_{lb}'] = (data.groupby(level=1).close.shift(lb) - data.groupby(
                    level=1).open.shift(lb)) / data.groupby(level=1).open.shift(lb)
                data[f'open_close_ratio_{lb}_pct'] = data[f'open_close_ratio_{lb}'].groupby(level=1).shift(lb).groupby(level=1).pct_change(1)
                data[f'low_high_ratio_{lb}'] = (data.groupby(level=1).high.shift(lb) - data.groupby(
                    level=1).low.shift(lb)) / data.groupby(level=1).low.shift(lb)
                data[f'low_high_ratio_{lb}_pct'] = data[f'low_high_ratio_{lb}'].groupby(level=1).shift(lb).groupby(level=1).pct_change(1)
                for f in ['open', 'high', 'low', 'close', 'volume']:
                    data[f'{f}_pct_{lb}'] = data.groupby(level=1).shift(lb).groupby(level=1)[f].pct_change(1)

            data['kmid'] = (data.close - data.open) / data.open
            data['klen'] = (data.high - data.low) / data.open
            data['kmid2'] = (data.close - data.open) / (data.high - data.low + 1e-12)
            data['kup'] = (data.high - np.maximum(data.open, data.close)) / data.open
            data['kup2'] = (data.high - np.maximum(data.open, data.close)) / (data.high - data.low + 1e-12)
            data['klow'] = (np.minimum(data.open, data.close) - data.low) / data.open
            data['klow2'] = (np.minimum(data.open, data.close) - data.low) / (data.high - data.low + 1e-12)
            data['ksft'] = (2 * data.close - data.high - data.low) / data.open
            data['ksft2'] = (2 * data.close - data.high - data.low) / (data.high - data.low + 1e-12)
            data['ratio_open_0'] = data.open / data.close
            data['ratio_high_0'] = data.high / data.close
            data['ratio_low_0'] = data.low / data.close

            # Open high low close volume ratios
            def ratio_price(df, shift):
                df[f"ratio_open_{shift}"] = df['open'].shift(shift) / df.close
                df[f"ratio_high_{shift}"] = df['high'].shift(shift) / df.close
                df[f"ratio_low_{shift}"] = df['low'].shift(shift) / df.close
                df[f"ratio_close_{shift}"] = df['close'].shift(shift) / df.close
                df[f"ratio_volume_close_{shift}"] = df['volume'].shift(shift) / df.close
                df[f"ratio_volume_volume_{shift}"] = df['volume'].shift(shift) / df.volume
                return df
            for w in ratio_range:
                res = [ratio_price(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def roc(df, w):
                df[f"roc_{w}"] = talib.ROC(df['close'], w)
                return df
            for w in windows:
                res = [roc(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def ma(df, w):
                df[f"ma_{w}"] = talib.MA(df.close, w) / df.close
                return df
            for w in windows:
                res = [ma(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def std(df, w):
                df[f"std_{w}"] = talib.STDDEV(df.close, w) / df.close
                return df
            for w in [x+1 for x in windows]:
                res = [std(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def beta(df, w):
                df[f"beta_{w}"] = talib.BETA(df.high, df.low, w) / df.close
                return df
            for w in windows:
                res = [beta(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def slope(df, w):
                df[f"slope_{w}"] = talib.LINEARREG_SLOPE(df.close, w) / df.close
                return df
            for w in [x+1 for x in windows]:
                res = [slope(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def max_high(df, w):
                df[f"max_high_{w}"] = (df.close.rolling(w).max()) / df.close
                return df
            for w in windows:
                res = [max_high(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def min_low(df, w):
                df[f"min_low_{w}"] = (df.close.rolling(w).min()) / df.close
                return df
            for w in windows:
                res = [min_low(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def qtlu(df, w):
                df[f"qtlu_{w}"] = (df.close.rolling(w).quantile(.8)) / df.close
                return df
            for w in windows:
                res = [qtlu(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def qtld(df, w):
                df[f"qtld_{w}"] = (df.close.rolling(w).quantile(.2)) / df.close
                return df
            for w in windows:
                res = [qtld(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def pctrank(x):
                i = x.argsort().argmax() + 1
                n = len(x)
                return i / n
            def rank(df, w):
                df[f"rank_{w}"] = df.close.rolling(w).apply(pctrank)
                return df
            for w in windows:
                res = [rank(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def rsv(df, w):
                roll_min = df.close.rolling(w).min()
                roll_max = df.close.rolling(w).max()
                df[f"rsv_{w}"] = (df.close - roll_min) / (roll_max - roll_min)
                return df
            for w in windows:
                res = [rsv(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def idx_max(df, w):
                df[f"idx_max_{w}"] = df.high.rolling(w).apply(np.argmax) / w
                return df
            for w in windows:
                res = [idx_max(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def idx_min(df, w):
                df[f"idx_min_{w}"] = df.low.rolling(w).apply(np.argmin) / w
                return df
            for w in windows:
                res = [idx_min(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def idx_mid(df, w):
                df[f"idx_mid_{w}"] = df.high.rolling(w).apply(np.argmax) - df.low.rolling(w).apply(np.argmin) / w
                return df
            for w in windows:
                res = [idx_mid(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def corr(df, w):
                df[f"corr_{w}"] = talib.CORREL(df.close, np.log(df.volume + 1), w)
                return df
            for w in windows:
                res = [corr(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def cord(df, w):
                df[f"cord_{w}"] = talib.CORREL((df.close / df.close.shift(1)), np.log(df.volume / df.volume.shift(1) + 1), w)
                return df
            for w in windows:
                res = [cord(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def cntp(df, w):
                df[f"cntp_{w}"] = talib.MA(df.close > df.close.shift(1), w)
                return df
            for w in windows:
                res = [cntp(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def cntn(df, w):
                df[f"cntn_{w}"] = talib.MA(df.close < df.close.shift(1), w)
                return df
            for w in windows:
                res = [cntn(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def cntd(df, w):
                df[f"cntd_{w}"] = talib.MA(df.close > df.close.shift(1), w) - talib.MA(df.close < df.close.shift(1), w)
                return df
            for w in windows:
                res = [cntd(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def sump(df, w):
                df[f"sump_{w}"] = (np.maximum(df.close - df.close.shift(1), 0)).rolling(w).sum() / (
                    np.abs(df.close - df.close.shift(1))).rolling(5).sum() + 1e-12
                return df
            for w in windows:
                res = [sump(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def sumn(df, w):
                df[f"sumn_{w}"] = (np.maximum(df.close.shift(1) - df.close, 0)).rolling(w).sum() / (
                    np.abs(df.close - df.close.shift(1))).rolling(5).sum() + 1e-12
                return df
            for w in windows:
                res = [sumn(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def sumd(df, w):
                df[f"sumd_{w}"] = ((np.maximum(df.close - df.close.shift(1), 0)).rolling(w).sum() -
                                    (np.maximum(df.close.shift(1) - df.close, 0)).rolling(w).sum()) / (
                                      np.abs(df.close - df.close.shift(1))).rolling(5).sum() + 1e-12
                return df
            for w in windows:
                res = [sumd(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def vma(df, w):
                df[f"vma_{w}"] = talib.MA(df.volume, w) / (df.volume + 1e-12)
                return df
            for w in windows:
                res = [vma(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def vstd(df, w):
                df[f"vstd_{w}"] = talib.STDDEV(df.volume, w) / (df.volume + 1e-12)
                return df
            for w in [x+1 for x in windows]:
                res = [vstd(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def wvma(df, w):
                df[f"wvma_{w}"] = talib.STDDEV(np.abs(df.close / df.close.shift(1) - 1) * df.volume) / (
                            talib.MA(np.abs(df.close / df.close.shift(1) - 1) * df.volume) + 1e-12)
                return df
            for w in [x+1 for x in windows]:
                res = [wvma(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def vsump(df, w):
                df[f"vsump_{w}"] = talib.STDDEV(np.abs(df.close / df.close.shift(1) - 1) * df.volume) / (
                            talib.MA(np.abs(df.close / df.close.shift(1) - 1) * df.volume) + 1e-12)
                return df
            for w in [x+1 for x in windows]:
                res = [vsump(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def vsumn(df, w):
                df[f"vsumn_{w}"] = np.maximum(df.volume.shift(1) - df.volume, 0).rolling(w).sum() / (
                            np.abs(df.volume - df.volume.shift(1)).rolling(w).sum() + 1e-12)
                return df
            for w in windows:
                res = [vsumn(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            def vsumd(df, w):
                df[f"vsumd_{w}"] = (np.maximum(df.volume - df.volume.shift(1), 0).rolling(w).sum() - np.maximum(
                    df.volume.shift(1) - df.volume, 0).rolling(w).sum()) / (
                                                np.abs(df.volume - df.volume.shift(1)).rolling(w).sum() + 1e-12)
                return df
            for w in windows:
                res = [vsumd(df, w) for sym, df in data.groupby(level=1)]
                data = pd.concat(res, axis=0)

            features = data.drop(columns=data_columns)
            features.replace([np.inf, -np.inf], 0, inplace=True)
            features = features.dropna(axis=1, thresh=int(((100 - 20) / 100) * features.shape[0] + 1))

            return features

        qlib_features = get_qlib(ohlcv.set_index(['date', 'ticker'])).sort_index().reset_index()

    if 'catch22' in feature_list:

        windows=(60, 90, 120, 150)

        columns = ['DN_HistogramMode_5','DN_HistogramMode_10','CO_f1ecac','CO_FirstMin_ac','CO_HistogramAMI_even_2_5',
                   'CO_trev_1_num','MD_hrv_classic_pnn40','SB_BinaryStats_mean_longstretch1','SB_TransitionMatrix_3ac_sumdiagcov',
                   'PD_PeriodicityWang_th0_01','CO_Embed2_Dist_tau_d_expfit_meandiff','IN_AutoMutualInfoStats_40_gaussian_fmmi',
                   'FC_LocalSimple_mean1_tauresrat','DN_OutlierInclude_p_001_mdrmd','DN_OutlierInclude_n_001_mdrmd',
                   'SP_Summaries_welch_rect_area_5_1','SB_BinaryStats_diff_longstretch0','SB_MotifThree_quantile_hh',
                   'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1','SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
                   'SP_Summaries_welch_rect_centroid','FC_LocalSimple_mean3_stderr']

        def rolling_c22(prices):
            features = []
            for w in windows:
                res = [pd.Series(catch22_all(c)['values']) for c in prices.close.rolling(w) if len(c) == w]
                res = pd.concat(res, axis=1).T
                res.columns = [f"{c}_{w}" for c in columns]
                res.index = prices.tail(len(res)).index
                res = res.reindex(prices.index)
                features.append(res)
            features = pd.concat(features, axis=1).sort_index()
            return features

        def get_catch22(prices):
            res = Parallel(n_jobs=cpu_count())(delayed(rolling_c22)(data) for ticker, data in prices.groupby('ticker'))
            features = pd.concat(res, axis=0).sort_index()
            return features

        catch_features = get_catch22(ohlcv.set_index(['date', 'ticker'])).reset_index()

    # merge
    if '86ta' in feature_list:
        if 'st' in feature_list:
            data = pd.concat([ohlcv_ta, ohlcv_st], axis=1).reset_index()
        else:
            data = ohlcv_ta.reset_index()
    else:
        data = ohlcv.reset_index(drop=True)

    if 'dd' in feature_list:
        data = data.merge(dd_data, on=['date', 'ticker'])

    if 'catch22' in feature_list:
        data = data.merge(catch_features, on=['date', 'ticker'])

    if 'qlib' in feature_list:
        data = data.merge(qlib_features, on=['date', 'ticker'])
    
    if 'hedging' in feature_list:
        data = data.merge(returns_mom, on=['date'])

    data = data.set_index(['date', 'ticker']).sort_index().reset_index()
    data = data.fillna(0)
    data = data.set_index('date')
    data = data[data.index.hour==8].reset_index()

    temp = data.set_index('date')
    train = temp.loc[train_start_date:train_end_date].reset_index()
    test = temp.loc[trade_start_date:trade_end_date].reset_index()

    train.index = train.date.factorize()[0]
    test.index = test.date.factorize()[0]

    tech_indicator_list = train.columns.tolist()[3:]

    return train, test, tech_indicator_list