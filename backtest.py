# Yajun Li 2019.7.10
##########################################################################################
from yajunli import linear_discrim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#
# direct_move, money_flow, minus_direct, plus_direct, plus_direct_move,
              stochastic, fast, relative_strength, ulti_osci,
# willer,
#  data_money, buy_sell_5m,
X = pd.concat([ strength, rate_ratio,
              rate_of_change, percent_oscillator,
               mom, macdfix, oscillator, data_sma,mmaa_5m_mean,
               flag, data_cp], axis=1)
X = X.replace([np.inf, -np.inf], np.nan)
y = y.replace([np.inf, -np.inf], np.nan)

X = X.fillna(method='backfill')
y = y.fillna(method='backfill')


def backtest(X, y, price_change, uu):

    earn = pd.DataFrame()
    correct_ratio = pd.DataFrame()
    d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for i in d:
        Xtrain = X.loc[uu[i]: uu[i+146]]
        # Xtrain = Xtrain[Xtrain.index[0:len(Xtrain) - 1]]
        Xtrain.drop(Xtrain.tail(1).index, inplace=True)

        yt = y.loc[uu[i]: uu[i+146]]
        yt = yt[yt.index[1:len(yt)]]

        Xpredt = X[X.index.date == uu[i + 146]]
        # Xpredt = Xpredt[Xpredt.index[0:len(Xpredt) - 1]]
        Xpredt.drop(Xpredt.tail(1).index, inplace=True)

        ylabel = y.loc[uu[i+146]:uu[i+147]]
        ylabel = ylabel[ylabel.index[1:len(ylabel)]]

        price_change_t = price_change.loc[uu[i+146]:uu[i+147]]
        # price_change_t = price_change_t[price_change_t.index[1:len(price_change_t)]]
        price_change_t.drop(price_change_t.head(1).index, inplace=True)

        # yfit = linear_discrim.linear_discrim(Xtrain, yt, Xpredt)

        clf = LinearDiscriminantAnalysis()
        clf.fit(Xtrain, yt)
        yfit = clf.predict(Xpredt)

        # correct ratio
        jj = yfit * ylabel
        func = lambda x: x+1 if x < 0 else x
        right = list(map(func, jj))
        correct_ratio_new = pd.DataFrame([sum(right)/len(yfit)])
        correct_ratio = pd.concat((correct_ratio, correct_ratio_new), axis=0)

        # earn
        earn_new = pd.DataFrame(yfit * price_change_t[1])
        earn = pd.concat((earn, earn_new), axis=0)

    return earn, correct_ratio

    np.mean(correct_ratio)

    cumulative = np.cumsum(earn)
    # remove the index of cumulative
    cumulative = cumulative.values
    # x = np.arange(0, len(cumulative), 1)
    x = np.linspace(0, 390, len(cumulative))

    plt.plot(cumulative, label='mmaa_5m_mean,  data_money, flag, data_cp', c='blue')

    cu1 = cumulative
    plt.plot(x, cu1, label="add willer")
    cu2 = cumulative
    plt.plot(x, cu2, label="remove willer")
    cu3 = cumulative
    plt.plot(x, cu3, label="train length 130 ")

    plt.xlabel('number of forecasts')
    plt.ylabel('net value')
    plt.title('Factor Comparison')
    plt.legend()
    plt.show()

    plt.savefig('600585_5min_1.png', transparent=False)
    plt.savefig('600585_1.png', transparent=True)



