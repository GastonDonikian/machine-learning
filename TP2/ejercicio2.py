import pandas as pd
import metrics


def exercise_a(data_frame):
    star_wordcount_tuples = list(zip(data_frame['Star Rating'].values, data_frame['wordcount'].values))
    filtered_list = list(filter(lambda x: x[0] == 1, star_wordcount_tuples))
    return sum([i[1] for i in filtered_list]) / len(filtered_list)


def exercise_b(data_frame):
    df_list = metrics.cross_validation(data_frame, 2)
    return df_list[0], df_list[1]


def preprocess_array(arr):
    title_sentiment = map(lambda x: 1 if x == 'positive' else 0 if x == 'negative' else x,
                          arr['titleSentiment'].values)
    word_count = arr['wordcount'].values
    sentiment_value = arr['sentimentValue'].values
    objective_variable = arr['Star Rating'].values
    explicative_variables = list(zip(word_count,
                                     title_sentiment,
                                     sentiment_value))
    variables = list(zip(objective_variable, explicative_variables))
    return variables


# classified_array:: (actual, knn, weighted knn)
def exercise_c(train, test):
    train_variables = preprocess_array(train)
    test_variables = preprocess_array(test)
    classified_array = []
    for i in test_variables:
        k_nearest = return_k_nearest(5, train_variables, i[1])
        classified_array.append((i[0], classify_neighbours(k_nearest), classify_weighted_neighbours(k_nearest)))
    return classified_array

# data_array:: (a,(b,c,d))
# value:: (b,c,d)
# returns:: [(a,b)]
def return_k_nearest(k: int, data_array, value):
    # x[1] -> (b,c,d)
    euclidean_array = list(map(lambda x: (x[0], euclidean_distance(x[1], value)), data_array))
    return sorted(euclidean_array, key=lambda x: x[1])[:k]


def classify_neighbours(k_neighbours):
    count_dict = {}
    for j in k_neighbours:
        if j[0] not in count_dict:
            count_dict[j[0]] = 0
        count_dict[j[0]] += 1
    return max(count_dict)


def classify_weighted_neighbours(k_neighbours):
    count_dict = {}
    for j in k_neighbours:
        if j[0] not in count_dict: count_dict[j[0]] = 0
        if j[1] != 0:
            count_dict[j[0]] += 1 / j[1]
    return max(count_dict)


# a:: (x,y,z) b::(x,y,z)
def euclidean_distance(a, b):
    ans = 0
    for i in range(len(a)):
        ans += pow(a[i] - b[i], 2)
    return pow(ans, 1 / 2)


def main():
    data_frame = pd.read_csv('./resources/reviews_sentiment.csv', delimiter=';')
    # print("Average word count for ratings of 1: " + str(round(exercise_a(data_frame), 2)))
    train, test = exercise_b(data_frame)
    print(exercise_c(train=train, test=test)[:10])


if __name__ == "__main__":
    main()
