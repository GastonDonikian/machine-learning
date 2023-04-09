import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import random
import metrics
import numpy as np
from collections import defaultdict

iterations = 10
SEED = 2023
random.seed(SEED)

categories = ['Nacional', 'Economia', 'Deportes', 'Salud']
most_common = 1000


#@profile
def filterUselessWordsAndTokenized(data):
    articles = ["la", "lo", "los", "las", "el", "ella", "ellos", "una", "unos", "un", "y", "al", "del", "le"]
    prepositions = ["a", "ante", "bajo", "cabe", "con", "contra",
                    "de", "desde", "durante", "en", "entre", "hacia",
                    "hasta", "mediante", "para", "por", "según",
                    "sin", "so", "sobre", "tras"]
    extra = ["que", "su", "se", "fue", "como", '|']
    data = data.lower()
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
                    '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '¡', '¿']
    tokenized_data = re.split('\W+', data)
    return list(filter(lambda x: x.lower() not in articles and
                                 x.lower() not in prepositions and
                                 x.lower() not in extra and
                                 x.lower() not in punctuations, tokenized_data))

#@profile
def count_words_in_array(title, category):
    word_count = 0
    for word in title.split():
        word_count += category.count(word)
    return word_count

def category_filter(df):

    filtered_df  = df[(df['categoria'] == 'Nacional') | (df['categoria'] == 'Economia') | (df['categoria'] == 'Deportes') | (df['categoria']== 'Salud')]
    return filtered_df

#@profile
def news_filter(data):
    dataFilter = []
    for l in data.values:
        if l[1] in categories:
            dataFilter.append(l)

    # train, test = train_test_split(dataFilter, test_size=0.3) #Preg si se puede usar esta lib

    train = data.to_numpy()
    nacional = list(filter(lambda x: x[-1] == 'Nacional', train))
    economia = list(filter(lambda x: x[-1] == 'Economia', train))
    deportes = list(filter(lambda x: x[-1] == 'Deportes', train))
    salud = list(filter(lambda x: x[-1] == 'Salud', train))

    nacional_filter = []
    economia_filter = []
    deportes_filter = []
    salud_filter = []

    for news in list(nacional):
        nacional_filter.append(filterUselessWordsAndTokenized(news[0]))

    for news in list(economia):
        economia_filter.append(filterUselessWordsAndTokenized(news[0]))

    for news in list(deportes):
        deportes_filter.append(filterUselessWordsAndTokenized(news[0]))

    for news in list(salud):
        salud_filter.append(filterUselessWordsAndTokenized(news[0]))

    return salud_filter, economia_filter, deportes_filter, nacional_filter


def most_commons_words_list(nacional_filter, economia_filter, salud_filter, deportes_filter):
    most_common_words_nacional = most_commons(nacional_filter)
    most_common_words_economia = most_commons(economia_filter)
    most_common_words_salud = most_commons(salud_filter)
    most_common_words_deportes = most_commons(deportes_filter)

    return most_common_words_deportes, most_common_words_economia, most_common_words_nacional, most_common_words_salud


def most_commons(words):
    word_counter = Counter()
    for n in words:
        word_counter.update(n)

    most_common_words = [word for word, count in word_counter.most_common(most_common)]

    return most_common_words

#@profile
def map_frecuency_words(words):
    dictionary = defaultdict(int)
    
    count = 0

    for t in words:
        for w in t:
            count+=1
            dictionary[w] +=  1

    return dictionary, count
  

def create_probability_dictonary(salud_filter, economia_filter, deportes_filter, nacional_filter):
    dict_deportes, len_deportes = map_frecuency_words(deportes_filter)
    dict_economia, len_economia = map_frecuency_words(economia_filter)
    dict_nacional, len_nacional = map_frecuency_words(nacional_filter)
    dict_salud, len_salud = map_frecuency_words(salud_filter)
    return dict_deportes,  dict_economia,  dict_nacional,  dict_salud, len_deportes, len_economia, len_nacional, len_salud
    

#@profile
def classify_input(title_input,  dict_deportes:dict,  dict_economia:dict,  dict_nacional:dict,  dict_salud:dict, len_deportes, len_economia, len_nacional, len_salud):
    # most_common_words_deportes, most_common_words_economia, most_common_words_nacional, most_common_words_salud = most_commons_words_list(nacional_filter, economia_filter, salud_filter, deportes_filter)


    words = set()
    words = words.union(set(dict_deportes.keys()))
    words = words.union(set(dict_economia.keys()))
    words = words.union(set(dict_nacional.keys()))
    words = words.union(set(dict_salud.keys()))



    count = len_deportes + len_economia + len_nacional + len_salud

    probability_deportes = len_deportes / count
    probability_economia = len_economia / count
    probability_nacional = len_nacional / count
    probability_salud = len_salud / count

    

    tokenized_data = title_input.split(' ')
    for word in words:
        if word in tokenized_data:
            probability_deportes *= ((dict_deportes.get(word, 0) + 1) / (len_deportes + 4))
            probability_economia *= ((dict_economia.get(word, 0) + 1) / (len_economia + 4))
            probability_nacional *= ((dict_nacional.get(word, 0) + 1) / (len_nacional + 4))
            probability_salud *= ((dict_salud.get(word, 0) + 1) / (len_salud + 4))
        else:
            probability_deportes *= 1-((dict_deportes.get(word, 0) + 1) / (len_deportes + 4))
            probability_economia *= 1-((dict_economia.get(word, 0) + 1) / (len_economia + 4))
            probability_nacional *= 1-((dict_nacional.get(word, 0) + 1) / (len_nacional + 4))
            probability_salud *= 1-((dict_salud.get(word, 0) + 1) / (len_salud + 4))

    total_prob = probability_deportes + probability_economia + probability_nacional + probability_salud

    dict = {}
    dict["Deportes"] = probability_deportes/total_prob
    dict["Economia"] = probability_economia/total_prob
    dict["Nacional"] = probability_nacional/total_prob
    dict["Salud"] = probability_salud/total_prob

    return dict

#@profile
def main():

    #training
    data = pd.read_excel('./resources/Noticias_argentinas.xlsx')
    data = pd.DataFrame(data, columns=['titular', 'categoria'])
    
    data = category_filter(data)


    training, test = metrics.cross_validation(data, 10)


    salud_filter, economia_filter, deportes_filter, nacional_filter = news_filter(training)
    dict_deportes,  dict_economia,  dict_nacional,  dict_salud, len_deportes, len_economia, len_nacional, len_salud = create_probability_dictonary(salud_filter, economia_filter, deportes_filter, nacional_filter)
  

   


    
    ######################################################################
    expected = test['categoria'].to_numpy()

    predicted = []
    #l = np.random.choice(test['titular'].to_numpy(),100)
    #testing
    for idx,i in enumerate(test['titular'].to_numpy()):
        print(round(idx/len(test['titular'].to_numpy())*100,2),end='\r')
        dictonary = classify_input(i, dict_deportes,  dict_economia,  dict_nacional,  dict_salud, len_deportes, len_economia, len_nacional, len_salud)
        category = max(dictonary, key=dictonary.get)
        #print(i,dictonary)
        predicted.append(category)
    


    confusion_matrix_expanded = metrics.confusion_matrix(categories, expected, predicted)
   
    for category in categories:
        confusion_matrix = metrics.confusion_matrix_by_category(category, expected,predicted)
        print(confusion_matrix)
        accurancy = metrics.accurancy(confusion_matrix)
        precision = metrics.precision(confusion_matrix)
        f1 = metrics.F1_score(confusion_matrix)
        print(category)
        print(accurancy)
        print(precision)
        print(f1)



if __name__ == "__main__":
    main()
    # classify_input("Trabajadores del Buenos Aires Design cortan la avenida Libertador por el cierre del centro comercial")
