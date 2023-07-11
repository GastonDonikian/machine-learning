import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pandas.plotting import parallel_coordinates
import seaborn as sns

def analyze(x):
    #function fit dbscan
    print("Started Analyzing")
    # cluster the data into five clusters
    dbscan = DBSCAN(eps = 0.3, min_samples = 4).fit(x) # fitting the model
    #labels = dbscan.labels_ # getting the labels
    #print("DBSCAN Result")
    #print(labels)

    column_names = x.columns.tolist()
    print(column_names)

    clusters = dbscan.fit_predict(x)
    

    # Get unique clusters, excluding noise points
    unique_clusters = np.unique(clusters)

    # Define a colormap for clusters
    colormap = plt.cm.get_cmap('viridis', len(unique_clusters) - 1)
    

    # Print the clusters
    #print("Data Points:\n", x)
    #print("Cluster Labels:\n", clusters)

    for colum in x.columns:
        # Create a copy of the dataframe and add the cluster labels
        ordered_df = data.copy()
        ordered_df['Cluster Labels'] = clusters
        # Sort the dataframe by the 'Calories' column
        ordered_df = ordered_df.sort_values(by=colum)
        # Create a line plot with different colors for each cluster
        for cluster in unique_clusters:
            if cluster == -1:
                print("HOLA")
                cluster_data = ordered_df[ordered_df['Cluster Labels'] == cluster]
                plt.scatter(cluster_data[colum], cluster_data['Cluster Labels'], marker='o', linestyle='', color='grey', label='Noise')
            else:
                cluster_data = ordered_df[ordered_df['Cluster Labels'] == cluster]
                plt.scatter(cluster_data[colum], cluster_data['Cluster Labels'], marker='o', linestyle='', color=colormap(cluster))



        plt.xlabel(colum)
        plt.ylabel('Cluster Labels')
        plt.title('Number of Clusters by ' + colum)
        plt.show()








    # for column in x.columns:
    #     plt.figure()
    #     plt.title(f'Heatmap of {column} by Cluster')
        
    #     # Create a copy of the dataframe and add the cluster labels
    #     ordered_df = x.copy()
    #     ordered_df['Cluster Labels'] = clusters
        
    #     # Sort the dataframe by the variable column within each cluster
    #     ordered_df = ordered_df.sort_values(by=[column])
    #     sns.heatmap(ordered_df[[column]], cmap='plasma', cbar=False)
    #     #plt.xlabel(clusters)
    #     y_labels = [round(value, 2) for value in ordered_df[column]]
    #     print(y_labels)
    #     #plt.ylabel(y_labels)
    #     plt.yticks(ticks=range(len(ordered_df)), labels=y_labels)
    #     plt.show()
    # Plot the clusters
    #plt.scatter(x[:, 0], x[:,1], c = labels, cmap= "plasma") # plotting the clusters
    #plt.scatter(x[column_names[0]], x[column_names[1]], c = labels, cmap= "plasma") # plotting the clusters
    #plt.xlabel("Avg Sleep Mins") # X-axis label
    #plt.ylabel("Avg Daily Steps") # Y-axis label
    #plt.show() # showing the plot


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
    analyze(normalized_x,x)
    


if __name__ == "__main__":
    preprocess_tables()
    