# Yajun Li 2019.7.9
# Generate attributes of all training and predict data with an interval of five minutes
# once and for all!
import pandas as pd
import numpy as np
import talib

def attribute_gen(data):
    '''
        generate attributes from cleansed data of local database
    :param data: is tick level np.array with 28 columns
    :return: X, y, Xpred are dataframe with datetime index
    '''
#######################################################################################
    # indexed by datetime
    data_indexed = pd.DataFrame(data)
    data_indexed = data_indexed.set_index(2)  # the original 28 column is removed automatically
    data_indexed.drop([0, 1], axis=1, inplace=True)  # 8503
    # type(data_indexed[7][0])  # it is int, we need float instead
    data_indexed[7] = data_indexed[7].astype(float)

#######################################################################################
    '''
    # five minutes first & last & mean
    # 1
    data_indexed_5m_first = data_indexed.resample('5T').first()
    data_indexed_5m_first = data_indexed_5m_first.between_time('9:30', '15:00')

    inde = data_indexed_5m_first[data_indexed_5m_first[3].isna()].index
    data_indexed_5m_first.drop(inde, inplace=True)
    # data_indexed_5m_first = data_indexed_5m_first.fillna(method='backfill')
    data_indexed_5m_first.drop(data_indexed_5m_first.tail(1).index, inplace=True)
    data_indexed_5m_first.drop(data_indexed_5m_first.tail(1).index, inplace=True)

    # 2
    data_indexed_5m_last = data_indexed.resample('5T').last()  # 354
    data_indexed_5m_last = data_indexed_5m_last.between_time('9:30', '15:00')  # 133
    inde = data_indexed_5m_last[data_indexed_5m_last[3].isna()].index
    data_indexed_5m_last.drop(inde, inplace=True)  # 96
    # data_indexed_5m_last = data_indexed_5m_last.fillna(method='backfill')
    # cut the last item
    data_indexed_5m_last.drop(data_indexed_5m_last.tail(1).index, inplace=True)

    # 3

    data_indexed_5m_last = data_indexed_5m_last.between_time('9:30', '15:00')  # 133
    inde = data_indexed_5m_last[data_indexed_5m_last[3].isna()].index
    data_indexed_5m_last.drop(inde, inplace=True)  # 96
    # data_indexed_5m_last = data_indexed_5m_last.fillna(method='backfill')
    # cut the last item
    data_indexed_5m_last.drop(data_indexed_5m_last.tail(1).index, inplace=True)

    # 4
    data_indexed_5m_change = data_indexed_5m_last - data_indexed_5m_first
    '''
########################################################################################
    # five minutes current price last & change & max & min
    data_cp = pd.DataFrame(data[:, [2, 3]])  # data is np.array()
    data_cp = data_cp.set_index(0)
    data_cplast = data_cp.resample('5T').last()  # 354
    inde = data_cplast[data_cplast[1].isna()].index
    data_cplast.drop(inde, inplace=True)  # 96

    # data_cpfirst = data_cp.resample('5T').first()  # 354
    # inde = data_cpfirst[data_cpfirst[1].isna()].index
    # data_cpfirst.drop(inde, inplace=True)  # 96

    data_cprice_change = data_cplast.diff()
    data_cprice_change = data_cprice_change.fillna(method='bfill')

    # data_cpratio = data_cplast/data_cpfirst
    # move one step forward
    # data_cprice_change = data_cprice_change[1][data_cprice_change.index[1:len(data_cprice_change)]]
    # cut the last item
    # data_cplast = data_cplast[1][data_cplast.index[0:len(data_cplast) - 1]]

    data_cpmax = data_cp.resample('5T').max()  # 354
    indexname = data_cpmax[data_cpmax[1].isna()].index
    data_cpmax.drop(indexname, inplace=True)  # 96
    # ?????? doesn't work: data_cpmax = data_cpmax.resample('5T').agg(lambda x: max(x[i]) for i in range(len(x)))  # 354
    # cut the last item
    # data_cpmax = data_cpmax[1][data_cpmax.index[0:len(data_cpmax)-1]]

    data_cpmin = data_cp.resample('5T').min()  # 354
    ind = data_cpmin[data_cpmin[1].isna()].index
    data_cpmin.drop(ind, inplace=True)  # 96
    # cut the last item
    # data_cpmin = data_cpmin[1][data_cpmin.index[0:len(data_cpmin)-1]]

    data_cp = pd.DataFrame()
    data_cp['max'] = data_cpmax[1]
    data_cp['min'] = data_cpmin[1]
    data_cp['last'] = data_cplast[1]
    data_cp['change'] = data_cprice_change[1]
    # data_cp['ratio'] = data_cpratio[1]  ratio can not increase forecast accuracy

######################################################################################################
    # flag data
    data_flag = pd.DataFrame(data[:, [2, 7]])
    data_flag = data_flag.set_index(0)
    data_flag_plus = data_flag
    data_flag_minus = data_flag

    # data_flag_plus[1][data_flag_plus[1] == -1] = 0
    # data_flag_minus[1][data_flag_minus[1] == 1] = 0
    # data_flag_minus[1][data_flag_minus[1] == -1] = 1

    # flagtry = list(map(lambda x: x+1 if x == -1 else x, data_flag_plus[1]))
    data_flag_plus = data_flag_plus.applymap(lambda x: x + 1 if x == -1 else x)
    data_flag_minus = data_flag_minus.applymap(lambda x: x - 1 if x == 1 else x + 2)

    data_flag_plsum = data_flag_plus.resample('5T').sum()  # 354
    data_flag_misum = data_flag_minus.resample('5T').sum()  # 354
    # index = data_flag_plsum[data_flag_plsum[1] == 0].index
    data_flag_plsum.drop(inde, inplace=True)  # 96
    data_flag_misum.drop(inde, inplace=True)  # 96
    data_flag_ratio = data_flag_plsum/data_flag_misum  # 96
    flag = pd.DataFrame()
    flag['psum'] = data_flag_plsum[1]
    # flag['msum'] = data_flag_misum[1]
    flag['ratio'] = data_flag_ratio[1]
    # cut the last item
    # flag.drop(flag.tail(1).index, inplace=True)

####################################################################################
    '''
    data_flag = pd.Series(data_flag[1])  # the index is attached automatically
    data_flag = data_flag.resample('5T').agg(lambda x: x[-1]/x[1] - 1)
    index = data_indexed_5m_first[data_flag[3].isna()].index
    data_flag.drop(index, inplace=True)
    data_flag = data_flag.fillna(method='backfill')
    '''
    # standardize money (column five)
    data_money = pd.DataFrame(data[:, [2, 5]])  # data is np.array()
    data_money = data_money.set_index(0)
    data_money = pd.Series(data_money[1])
    data_money_5m = data_money.resample('5T').sum()  # 354
    ind = data_money_5m[data_money_5m == 0.0].index
    data_money_5m.drop(ind, inplace=True)  # 96
    data_money_5m_d = data_money_5m.diff()
    data_money_5m_d = data_money_5m_d.fillna(method='backfill')
    # cut the last item
    # data_money_5m_d.drop(data_money_5m_d.tail(1).index, inplace=True)
    data_money = pd.DataFrame()
    data_money['money'] = data_money_5m
    data_money['diff'] = data_money_5m_d

#########################################################################################################
    # MACD
    macd = data_indexed[3]  # 8503
    macd = pd.DataFrame(macd)
    macd_12_ema = macd.ewm(span=12, adjust=False).mean()
    macd_26_ema = macd.ewm(span=26, adjust=False).mean()
    macd_real = macd_12_ema - macd_26_ema
    macd_reference = macd_real.ewm(span=9, adjust=False).mean()
    # transform to 5 minutes' frequency
    mmaa = pd.DataFrame()
    mmaa['12'] = macd_12_ema[3]
    mmaa['26'] = macd_26_ema[3]
    mmaa['real'] = macd_real[3]
    mmaa['refer'] = macd_reference[3]
    mmaa_5m = mmaa.resample('5T')
    mmaa_5m_mean = mmaa_5m.mean()  # 354
    indd = mmaa_5m_mean[mmaa_5m_mean['12'].isna()].index
    mmaa_5m_mean.drop(indd, inplace=True)  # 96
    # cut the last item
    # mmaa_5m_mean.drop(mmaa_5m_mean.tail(1).index, inplace=True)  # 95

    # macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12,
    #                                         slowperiod=26, signalperiod=9)

####################################################################################
    high = data_cpmax[1]
    low = data_cpmin[1]
    close = data_cplast[1]

    data_vol = pd.DataFrame(data[:, [2, 6]])  # data is np.array()
    data_vol = data_vol.set_index(0)
    data_vol = pd.Series(data_vol[1])
    data_vol_5m = data_vol.resample('5T').sum()  # 354
    ind = data_vol_5m[data_vol_5m == 0.0].index
    data_vol_5m.drop(ind, inplace=True)  # 96
    volume = data_vol_5m
####################################################################################
    # sma of current price
    data_sma_20 = talib.SMA(np.asarray(close), 20)
    data_sma_20 = pd.DataFrame(data_sma_20)
    data_sma_20 = data_sma_20.fillna(method='backfill')

    data_sma_10 = talib.SMA(np.asarray(close), 10)
    data_sma_10 = pd.DataFrame(data_sma_10)
    data_sma_10 = data_sma_10.fillna(method='backfill')

    data_sma_5 = talib.SMA(np.asarray(close), 5)
    data_sma_5 = pd.DataFrame(data_sma_5)
    data_sma_5 = data_sma_5.fillna(method='backfill')

    data_sma = pd.DataFrame()
    data_sma['5'] = data_sma_5[0]
    data_sma['10'] = data_sma_10[0]
    data_sma['20'] = data_sma_20[0]
    data_sma['0'] = volume.index
    data_sma = data_sma.set_index('0')

####################################################################################
    # Average Directional Movement Index
    # real = talib.ADX(high, low, close, timeperiod=14)
    direct_move = talib.ADX(high, low, close, timeperiod=14)
    direct_move = direct_move.fillna(method='backfill')

########################################################################################
    # Absolute Price Oscillator
    oscillator = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    oscillator = oscillator.fillna(method='backfill')

########################################################################################
    # macd fix
    macdfix, macdsignal, macdhist = talib.MACDFIX(close, signalperiod=9)  # nan 40
    macdfix.fillna(method='backfill')

#########################################################################################
    # money flow
    money_flow = talib.MFI(high, low, close, volume, timeperiod=14)
    money_flow.fillna(method='backfill')

#########################################################################################
    # minus directional indicator
    minus_direct = talib.MINUS_DI(high, low, close, timeperiod=14)
    minus_direct.fillna(method='backfill')

#########################################################################################
    # momentum of the close prices, with a time period of 5:
    mom = talib.MOM(close, timeperiod=10)
    mom = mom.fillna(method='backfill')

#########################################################################################
    # PLUS_DI - Plus Directional Indicator
    plus_direct = talib.PLUS_DI(high, low, close, timeperiod=14)
    plus_direct = plus_direct.fillna(method='backfill')

#########################################################################################
    # PLUS_DM - Plus Directional Movement
    plus_direct_move = talib.PLUS_DM(high, low, timeperiod=14)
    plus_direct_move = plus_direct_move.fillna(method='backfill')

#########################################################################################
    # PPO - Percentage Price Oscillator
    percent_oscillator = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    percent_oscillator = percent_oscillator.fillna(method='backfill')

#########################################################################################
    # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
    rate_of_change = talib.ROCP(close, timeperiod=10)
    rate_of_change = rate_of_change.fillna(method='backfill')

#########################################################################################
    # Rate of change ratio
    rate_ratio = talib.ROCR(close, timeperiod=10)
    rate_ratio = rate_ratio.fillna(method='backfill')

#########################################################################################
    # RSI - Relative Strength Index
    strength = talib.RSI(close, timeperiod=14)
    strength = strength.fillna(method='backfill')

#########################################################################################
    # STOCH - Stochastic
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3,
                               slowd_matype=0)
    slowk = slowk.fillna(method='backfill')
    slowd = slowd.fillna(method='backfill')
    stochastic = pd.DataFrame()
    stochastic['slowk'] = slowk
    stochastic['slowd'] = slowd

#########################################################################################
    # STOCHF - Stochastic Fast
    fastk, fastd = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastk = fastk.fillna(method='backfill')
    fastd = fastd.fillna(method='backfill')
    fast = pd.DataFrame()
    fast['fastk'] = fastk
    fast['fastd'] = fastd

#########################################################################################
    # STOCHRSI - Stochastic Relative Strength Index
    fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastk = fastk.fillna(method='backfill')
    fastd = fastd.fillna(method='backfill')
    relative_strength = pd.DataFrame()
    relative_strength['fastk'] = fastk
    relative_strength['fastd'] = fastd

#########################################################################################
    # ULTOSC - Ultimate Oscillator
    ulti_osci = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    ulti_osci = ulti_osci.fillna(method='backfill')

#########################################################################################
    # WILLR - Williams' %R
    willer = talib.WILLR(high, low, close, timeperiod=14)
    willer = willer.fillna(method='backfill')

#########################################################################################
    # buy sell info factors
    buy_sell = data[:, 8:28]
    buy_sell = pd.DataFrame(buy_sell, index=data[:, 2])
    # col 8-12 buy price, col 13-17 sell price,
    # col 18-22 buy quant, col 23-27 sell quant
    buy_sell_5m = buy_sell.resample('5T').last()  # 354
    inde = buy_sell_5m[buy_sell_5m[1].isna()].index
    buy_sell_5m.drop(inde, inplace=True)  # 96

#########################################################################################
    # bollinger bands, with triple exponential moving average:
    from talib import MA_Type
    upper, middle, lower = talib.BBANDS(close, matype=MA_Type.T3)
#########################################################################################
    # generate label y
    data_cp2 = pd.DataFrame(data[:, [2, 3]])  # data is np.array()
    data_cp2 = data_cp2.set_index(0)
    data_cplast = data_cp2.resample('5T').last()  # 354
    inddd = data_cplast[data_cplast[1].isna()].index
    data_cplast.drop(inddd, inplace=True)  # 96
    pchange = data_cplast.diff()  # 96
    pchange = pchange.fillna(method='backfill')

    # cut the last item
    # data_cplast = data_cplast[1][data_cplast.index[0:len(data_cplast) - 1]]

    func = lambda x: np.sign(x) if x != 0 else np.sign(x+1)
    pchange['sign'] = list(map(func, pchange[1]))
    # pchange.drop(pchange.head(1).index, inplace=True)  # 95
####################################################################################
    # generate train and label data
    X = pd.concat([mmaa_5m_mean, data_sma, direct_move, oscillator, macdfix, money_flow, minus_direct,
                   mom, plus_direct, plus_direct_move, percent_oscillator, rate_of_change,
                   rate_ratio, strength, stochastic, fast, relative_strength, ulti_osci,
                   willer, data_money, flag, data_cp], axis=1)
    y = pchange['sign']
    price_change = data_cprice_change

    return X, y, price_change








