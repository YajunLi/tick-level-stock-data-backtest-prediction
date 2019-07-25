import pandas as pd
import numpy as np


def get_all_data(stock_data):

    # returns a numpy array
        data = pd.DataFrame(stock_data)
        tradetime = pd.to_datetime(data[2])
        T = [tradetime.dt.hour, tradetime.dt.minute, tradetime.dt.second]
        T = np.array(T).transpose()
        B = np.array([[60, 0], [1, 0], [0, 1]])
        A = np.dot(T, B)
        A[:, [0]] = A[:, [0]] - 570

        func = lambda x: x[0] - 89 if x[0] > 120 else x[0]
        A[:, [0]] = np.array(list(map(func, A))).reshape(len(A), 1)

        # data cleaning
        data = np.array(data)
        indices = [i for (i, v) in enumerate(A[:, [0]]) if v < 0 or v > 240]
        data = np.delete(data, indices, 0)  # 0 means to delete rows, 1 columns
        A = np.delete(A, indices, 0)
        # flag to 0 or 1
        data[:, 7][data[:, 7] == 'B'] = 1
        data[:, 7][data[:, 7] == 'S'] = -1

        tradetime = tradetime.drop(indices)
        uu = tradetime.map(lambda t: t.date()).unique()

        return data, uu, A, tradetime





