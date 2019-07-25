import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def linear_discrim(X, y, Xpredt):
    '''
    linear discriminant classifier
    '''
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    X = X.fillna(method='backfill')
    y = y.fillna(method='backfill')

    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    yfit = clf.predict(Xpredt)

    return yfit
