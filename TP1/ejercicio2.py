import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

categories = ['Nacional', 'Destacadas', 'Deportes', 'Salud']
most_common = 100

def filterUselessWordsAndTokenized(data):
    articles = ["la","lo","los","las","el","ella","ellos","una","unos","un","y","al","del","le"]
    prepositions = ["a", "ante", "bajo", "cabe", "con", "contra", 
                    "de", "desde", "durante", "en","entre", "hacia",
                    "hasta", "mediante", "para", "por", "según", 
                    "sin", "so", "sobre", "tras"]
    extra = ["que", "su", "se", "fue", "como"]
    tokenized_data = data.split(' ')
    return list(filter(lambda x: x.lower() not in articles and
                        x.lower() not in prepositions and
                        x.lower() not in extra,tokenized_data))
    
def count_words_in_array(title, category):
    word_count = 0
    for word in title.split():
        word_count += category.count(word)
    return word_count

def news():
    data = pd.read_excel('./resources/Noticias_argentinas.xlsx')
    data = pd.DataFrame(data, columns = ['titular', 'fuente', 'categoria']) 
    dataFilter = []
    for l in data.values:
        if l[2] in categories:
            dataFilter.append(l)
    
    train, test = train_test_split(dataFilter, test_size=0.3) #Preg si se puede usar esta lib
    

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
    
    word_counter = Counter()
    for n in nacional_filter:
        word_counter.update(n)
    
    most_common_words_nacional = [word for word, count in word_counter.most_common(most_common)]
   
    for n in destacadas_filter:
        word_counter.update(n)
    
    most_common_words_destacadas = [word for word, count in word_counter.most_common(most_common)]

    for n in salud_filter:
        word_counter.update(n)
    
    most_common_words_salud = [word for word, count in word_counter.most_common(most_common)]

    for n in deportes_filter:
        word_counter.update(n)
    
    most_common_words_deportes = [word for word, count in word_counter.most_common(most_common)]

    return salud_filter, destacadas_filter, deportes_filter, nacional_filter


def classify_input(title_input):
     
    tokenized_data = title_input.split(' ')


if __name__ == "__main__":
    #print(classify_input([1,0,1,1,0])) 
    #print(classify_input([0,1,1,0,1])) 
    news()
