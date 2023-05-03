import numpy as np
import pandas as pd
import metrics
import matplotlib.pyplot as plt


def exercise_a(data_frame):
    star_wordcount_tuples = list(zip(data_frame['Star Rating'].values, data_frame['wordcount'].values))
    filtered_list = list(filter(lambda x: x[0] == 1, star_wordcount_tuples))
    return sum([i[1] for i in filtered_list]) / len(filtered_list)


def exercise_b(data_frame):
    df_list = metrics.cross_validation_for_2(data_frame, 5)
    train = pd.concat([df_list[1], df_list[2], df_list[3], df_list[4]], axis=0)
    test = df_list[0]
    return train, test


def preprocess_data_frame(df):
    df = df.dropna()
    df['titleSentiment'] = standarize_array(scale_array(list(map(lambda x: 1 if x == 'positive' else 0 if x == 'negative' else x,
                                  df['titleSentiment'].values))))
    df['wordcount'] = standarize_array(scale_array(df['wordcount'].values))
    df['sentimentValue'] = standarize_array(scale_array(df['sentimentValue'].values))
    return df


def preprocess_array(arr):
    title_sentiment = arr['titleSentiment'].values
    word_count = arr['wordcount'].values
    sentiment_value = arr['sentimentValue'].values
    objective_variable = arr['Star Rating'].values
    explicative_variables = list(zip(word_count,
                                     title_sentiment,
                                     sentiment_value))
    variables = list(zip(objective_variable, explicative_variables))
    return variables


def standarize_array(values):
    average_value = sum(values) / len(values)
    std_value = np.std(values)
    return list(map(lambda x: (x - average_value) / std_value, values))


def scale_array(values):
    max_value = max(values)
    min_value = min(values)
    range_value = max_value - min_value

    return list(map(lambda x: (x - min_value) / range_value, values))


# classified_array:: (actual, knn, weighted knn)
def exercise_c(train, test):
    classified_array = []
    for i in test:
        k_nearest = return_k_nearest(5, train, i[1])
        classified_array.append((i[0], classify_neighbours(k_nearest), classify_weighted_neighbours(k_nearest)))
    return classified_array


# data_array:: (a,(b,c,d))
# value:: (b,c,d)
# returns:: [(a,b)]
def return_k_nearest(k: int, data_array, value):
    # x[1] -> (b,c,d)
    euclidean_array = list(map(lambda x: (x[0], euclidean_distance(x[1], value)), data_array))
    ans = sorted(euclidean_array, key=lambda x: x[1])[:k]
    return ans


##return the most common class of the k_netbours counting them
def classify_neighbours(k_neighbours):
    count_dict = {}
    for j in k_neighbours:
        if j[0] not in count_dict:
            count_dict[j[0]] = 0
        count_dict[j[0]] += 1
    return max(count_dict, key=count_dict.get)


##return the class with the highest weighted score of the k_neighbours
def classify_weighted_neighbours(k_neighbours):
    count_dict = {}
    for j in k_neighbours:
        if j[0] not in count_dict: count_dict[j[0]] = 0
        if j[1] != 0:
            count_dict[j[0]] += 1 / pow(j[1], 2)
        else:
            count_dict[j[0]] += np.inf
    return max(count_dict, key=count_dict.get)


# a:: (x,y,z) b::(x,y,z)
def euclidean_distance(a, b):
    ans = 0
    for i in range(len(a)):
        ans += pow(a[i] - b[i], 2)
    return pow(ans, 1 / 2)


def exercise_d(k, train, test, weighted=True):
    train_variables = train
    test_variables = test
    expected = []
    predicted = []
    for i in test_variables:
        k_nearest = return_k_nearest(k, train_variables, i[1])
        expected.append(i[0])
        if weighted:
            predicted.append(classify_weighted_neighbours(k_nearest))
        else:
            predicted.append(classify_neighbours(k_nearest))
    # print_confusion_matrix(predicted=predicted, expected=expected)
    return expected, predicted


def print_confusion_matrix(predicted, expected):
    for i in range(1, 6):
        confusion_matrix, tasa_falsos_positivos, tasa_verdaderos_postivos = \
            metrics.confusion_matrix_by_category(category=i,
                                                 predicted=predicted,
                                                 expected=expected)
        print("category: " + str(i))
        print("Precision: " + str(metrics.precision(confusion_matrix)))
    confusion_matrix= \
        metrics.confusion_matrix(classes=[1,2,3,4,5],
                                             predicted=predicted,
                                             expected=expected)
    metrics.plot_confusion_matrix(confusion_matrix)


def plot_acc_vs_k_n(data_frame):
    accuracy_points_w = []
    std_points_w = []

    accuracy_points = []
    std_points = []
    k_points = []
    repetitions = 100
    standarized_data_frame = preprocess_data_frame(data_frame)
    for k in range(3, 50, 2):
        accuracy_per_iter_w = []
        accuracy_per_iter = []
        for i in range(repetitions):
            train, test = exercise_b(standarized_data_frame)
            train = preprocess_array(train)
            test = preprocess_array(test)

            expected, predicted = exercise_d(k=k, train=train, test=test, weighted=True)
            accuracy_per_iter_w.append(metrics.calculate_accuracy(expected, predicted))
            expected, predicted = exercise_d(k=k, train=train, test=test, weighted=False)
            accuracy_per_iter.append(metrics.calculate_accuracy(expected, predicted))

        accuracy_points_w.append(np.average(accuracy_per_iter_w))
        std_points_w.append(np.std(accuracy_per_iter_w) / np.sqrt(repetitions))
        accuracy_points.append(np.average(accuracy_per_iter))
        std_points.append(np.std(accuracy_per_iter) / np.sqrt(repetitions))
        k_points.append(k)
    plt.errorbar(k_points, accuracy_points_w, yerr=std_points_w, capsize=2, elinewidth=0.5, label="Weighted KNN")
    plt.errorbar(k_points, accuracy_points, yerr=std_points, capsize=2, elinewidth=0.5, label="KNN")

    plt.title('Accuracy vs k-neighbours over ' + str(repetitions) + ' repetitions.')
    plt.legend()
    plt.xlabel('k neighbours')
    plt.ylabel('accuracy')
    plt.savefig('./images/knn.png', bbox_inches='tight')
    plt.show()

def main():
    data_frame = pd.read_csv('./resources/reviews_sentiment.csv', delimiter=';')
    print("Average word count for ratings of 1: " + str(round(exercise_a(data_frame), 2)))
    standarized_data_frame = preprocess_data_frame(data_frame)
    train, test = exercise_b(standarized_data_frame)
    train = preprocess_array(train)
    test = preprocess_array(test)
    print_confusion_matrix(*exercise_d(k=7, train=train, test=test))

    # print(exercise_c(train=train, test=test)[:10])
    # plot_acc_vs_k_n(data_frame=data_frame)


if __name__ == "__main__":
    main()
