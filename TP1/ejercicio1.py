import pandas as pd

def classifier():
    data = pd.read_excel('./resources/PreferenciasBritanicos.xlsx')
    people = []

    for l in data.values:
       people.append(l)
    
    scottish = list(filter(lambda x: x[-1] == 'E', people))
    english = list(filter(lambda x: x[-1] == 'I', people))
    probabilities_given_scottish = []
    probabilities_given_english = []
    probabilities = []
    for i in range(len(data.columns) - 1):
        probabilities_given_scottish.append(len(list(filter(lambda x: x[i] == 1,scottish)))/len(scottish))
        probabilities_given_english.append(len(list(filter(lambda x: x[i] == 1,english)))/len(english))
        probabilities.append(len(list(filter(lambda x: x[i] == 1,people)))/len(people))

    probScottish = len(scottish)/len(people)
    probEnglish = len(english)/len(people)  
    return probabilities, probabilities_given_english, probabilities_given_scottish,probEnglish,probScottish
            
def classify_input(qualities):
    probabilities, probabilities_given_english, probabilities_given_scottish, probabilityEnglish, probabilityScottish = classifier()
    for idx,i in enumerate(qualities):
        if i == 1:
            probabilityEnglish = probabilityEnglish*(probabilities_given_english[idx])
            probabilityScottish = probabilityScottish*(probabilities_given_scottish[idx])
        else:
            probabilityEnglish = probabilityEnglish*(1-probabilities_given_english[idx])
            probabilityScottish = probabilityScottish*(1-probabilities_given_scottish[idx])

    return probabilityScottish, probabilityEnglish 





if __name__ == "__main__":
    print(classify_input([1,0,1,1,0]))
    print(classify_input([0,1,1,0,1])) ##chequear que uno da 0

    

