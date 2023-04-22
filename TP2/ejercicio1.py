import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import random
import metrics
import numpy as np
from collections import defaultdict


def preprocessing(df):
    df['Duration of Credit (month)'] ##finish this
    return df








if __name__ == "__main__":
    data = pd.read_csv('./resources/german_credit.csv')
    data = preprocessing(data)
    df_list = metrics.cross_validation(data, 10)
    test = df_list[0]
    training = pd.DataFrame()
    for j in range(1, 10):
        training = pd.concat([training, df_list[j]], axis=0)
