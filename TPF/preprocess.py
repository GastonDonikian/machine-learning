import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def analyze(x):
    #function fit dbscan
    print("Started Analyzing")
    # cluster the data into five clusters
    dbscan = DBSCAN(eps = 0.3, min_samples = 2).fit(x) # fitting the model
    labels = dbscan.labels_ # getting the labels
    print("DBSCAN Result")
    print(labels)

    column_names = x.columns.tolist()
    print(column_names)
    # Plot the clusters
    #plt.scatter(x[:, 0], x[:,1], c = labels, cmap= "plasma") # plotting the clusters
    plt.scatter(x[column_names[0]], x[column_names[1]], c = labels, cmap= "plasma") # plotting the clusters
    plt.xlabel("Avg Sleep Mins") # X-axis label
    plt.ylabel("Avg Daily Steps") # Y-axis label
    plt.show() # showing the plot


def preprocess_tables():

    daily_sleep_mins_per_user_id = pd.read_csv('./resources/sleepDay_merged.csv', delimiter=',')
    daily_sleep_mins_per_user_id = daily_sleep_mins_per_user_id.dropna()

    daily_sleep_mins_per_user_id = daily_sleep_mins_per_user_id.rename(columns={'SleepDay': 'Date'})
    daily_sleep_mins_per_user_id['Date'] = daily_sleep_mins_per_user_id['Date'].apply(lambda x: x.split(" ")[0])

    daily_activity_per_user_id = pd.read_csv('./resources/dailyActivity_merged.csv', delimiter=',')
    daily_activity_per_user_id = daily_activity_per_user_id.dropna()

    daily_activity_per_user_id = daily_activity_per_user_id.rename(columns={'ActivityDate': 'Date'})
    
    print("Finished Preprocessing")
    
    #x = avg_sleep_mins_per_user_id.set_index('Id').join(avg_daily_steps_per_user_id.set_index('Id')).join(avg_daily_distance_per_user_id.set_index('Id'))
    x = pd.merge(daily_sleep_mins_per_user_id,daily_activity_per_user_id , on=['Id', 'Date'], how='inner')

  
    x = x[['Calories', 'TotalSteps', 'TotalDistance', 'TotalMinutesAsleep']]
    print(x)
    print(x.size)
   
    normalized_x=(x-x.mean())/x.std()
    analyze(normalized_x)
    


if __name__ == "__main__":
    preprocess_tables()
    