import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import random
import metrics

iterations = 10
SEED = 2023
random.seed(SEED)

categories = ['Nacional', 'Destacadas', 'Deportes', 'Salud']
most_common = 1000

def filterUselessWordsAndTokenized(data):
    articles = ["la","lo","los","las","el","ella","ellos","una","unos","un","y","al","del","le"]
    prepositions = ["a", "ante", "bajo", "cabe", "con", "contra", 
                    "de", "desde", "durante", "en","entre", "hacia",
                    "hasta", "mediante", "para", "por", "según", 
                    "sin", "so", "sobre", "tras"]
    extra = ["que", "su", "se", "fue", "como",'|']
    data = data.lower()
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '¡', '¿']
    tokenized_data = re.split('\W+',data)
    return list(filter(lambda x: x.lower() not in articles and
                        x.lower() not in prepositions and
                        x.lower() not in extra and
                        x.lower() not in punctuations,tokenized_data))
    
def count_words_in_array(title, category):
    word_count = 0
    for word in title.split():
        word_count += category.count(word)
    return word_count

def news_filter(data):
   
    dataFilter = []
    for l in data.values:
        if l[2] in categories:
            dataFilter.append(l)
    
   # train, test = train_test_split(dataFilter, test_size=0.3) #Preg si se puede usar esta lib
    
    train = data.to_numpy()
    nacional= list(filter(lambda x: x[-1] == 'Nacional',train))
    destacadas= list(filter(lambda x: x[-1] == 'Destacadas',train))
    deportes= list(filter(lambda x: x[-1] == 'Deportes',train))
    salud = list(filter(lambda x: x[-1] == 'Salud',train))
    

    nacional_filter = []
    destacadas_filter = []
    deportes_filter = []
    salud_filter = []
    
    for news in list(nacional):
        nacional_filter.append(filterUselessWordsAndTokenized(news[0]))

    for news in list(destacadas):
        destacadas_filter.append(filterUselessWordsAndTokenized(news[0]))
    
    for news in list(deportes):
        deportes_filter.append(filterUselessWordsAndTokenized(news[0]))

    for news in list(salud):
        salud_filter.append(filterUselessWordsAndTokenized(news[0]))
    

    return salud_filter, destacadas_filter, deportes_filter, nacional_filter

def most_commons_words_list(nacional_filter, destacadas_filter, salud_filter, deportes_filter):
    most_common_words_nacional = most_commons(nacional_filter)
    most_common_words_destacadas = most_commons(destacadas_filter)
    most_common_words_salud = most_commons(salud_filter)
    most_common_words_deportes = most_commons(deportes_filter)

    return most_common_words_deportes, most_common_words_destacadas, most_common_words_nacional, most_common_words_salud

def most_commons(words):
    word_counter = Counter()
    for n in words:
        word_counter.update(n)
    
    most_common_words = [word for word, count in word_counter.most_common(most_common)]

    return most_common_words

def map_frecuency_words(words): 
    dictonary = {}
    count = 0
    for t in words:
        for w in t:
            count += 1
            
    for t in words:
        for w in t:
            dictonary[w] = dictonary.setdefault(w, 0) + (1/len(words)) 
    
    return dictonary

def classify_input(title_input,salud_filter, destacadas_filter, deportes_filter, nacional_filter):

   
    #most_common_words_deportes, most_common_words_destacadas, most_common_words_nacional, most_common_words_salud = most_commons_words_list(nacional_filter, destacadas_filter, salud_filter, deportes_filter)
    dict_deportes = map_frecuency_words(deportes_filter)
    dict_destacadas = map_frecuency_words(destacadas_filter)
    dict_nacional = map_frecuency_words(nacional_filter)
    dict_salud = map_frecuency_words(salud_filter)

    len_deportes = len(deportes_filter)
    len_destacadas = len(destacadas_filter)
    len_nacional = len(nacional_filter)
    len_salud = len(salud_filter)
    count = len_deportes + len_destacadas + len_nacional + len_salud
    
    probability_deportes = len_deportes / count
    probability_destacadas = len_destacadas / count
    probability_nacional = len_nacional / count
    probability_salud = len_salud / count
    
    
    
    tokenized_data = title_input.split(' ')


    for word in tokenized_data:
        probability_deportes *=  (( dict_deportes.get(word, 0) + 1)/4)
        probability_destacadas *=  ((dict_destacadas.get(word,  0) + 1)/4)
        probability_nacional *=  ((dict_nacional.get(word,  0) + 1)/4)
        probability_salud *=  ((dict_salud.get(word,  0) + 1)/4)

    dict = {}
    dict["deportes"] = probability_deportes
    dict["destacadas"] = probability_destacadas
    dict["nacional"] = probability_nacional
    dict["salud"] = probability_salud

    return dict

    



if __name__ == "__main__":
    data = pd.read_excel('./resources/Noticias_argentinas.xlsx')
    data = pd.DataFrame(data, columns = ['titular', 'fuente', 'categoria']) 

    training, test = metrics.cross_validation(data,10)

    salud_filter, destacadas_filter, deportes_filter, nacional_filter = news_filter(training)
   
    expected = training['categoria'].to_numpy()
    

    predicted = []
    
    for i in training['titular'].to_numpy():
        dict = classify_input(i,salud_filter, destacadas_filter, deportes_filter, nacional_filter)
        category = max(dict, key=dict.get)      
        predicted.append(category)



    metrics.confusion_matrix(4,expected,predicted)
    #classify_input("Trabajadores del Buenos Aires Design cortan la avenida Libertador por el cierre del centro comercial")
