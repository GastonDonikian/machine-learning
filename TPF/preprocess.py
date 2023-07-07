import pandas as pd


# A nosotros nos interesa mergear
# dailyActivity_merged.csv con sleepDay_merged.csv
def join_sleep_with_activity():
    data_frame_sleep = pd.read_csv('./resources/sleepDay_merged.csv')
    data_frame_activity = pd.read_csv('./resources/dailyActivity_merged.csv')
