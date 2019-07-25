# Yajun Li 2019.7.11
####################################################################
# access the data from local database
import psycopg2
from yajunli import get_table_col_names
from yajunli import get_all_data
from yajunli import attribute_gen
from yajunli import backtest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

con = psycopg2.connect("dbname='BrooksCapital' host = '192.168.31.94' port='5432' user='julin'  password='123456'")
cur = con.cursor()
cur.execute("""SELECT * FROM stocktick
       WHERE c_stock_code = '603899'
       AND c_date_time BETWEEN '2018-06-01' AND '2019-08-30'""")
stock_data = cur.fetchall()

# get the data column names
colnames = get_table_col_names(con, 'stocktick')


def main():

    drawer = get_all_data(stock_600585)
    data = drawer[0]
    uu = drawer[1]

    drawer2 = attribute_gen(data)
    X = drawer2[0]
    y = drawer2[1]
    price_change = drawer2[2]

    result = backtest(X, y, price_change, uu)
    earn = result[0]
    correct_ratio = result[1]

    # average
    np.mean(correct_ratio)

    # plot
    values, base = np.histogram(earn, bins=40)
    # evaluate the cumulative
    cumulative = np.cumsum(values)
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c='blue')
    plt.show()

    # second thought plot
    cumulative = np.cumsum(earn)
    # remove the index of cumulative

    cumulative = cumulative.values
    plt.plot(cumulative, c='blue')
    plt.show()


if __name__ == '__main__':
    main()


