import pandas as pd

k = 2


def classifier():
    data = pd.read_excel('./resources/PreferenciasBritanicos.xlsx')
    people = []

    for l in data.values:
        people.append(l)

    scottish = list(filter(lambda x: x[-1] == 'E', people))
    english = list(filter(lambda x: x[-1] == 'I', people))
    probabilities_given_scottish = []
    probabilities_given_english = []

    for i in range(len(data.columns) - 1):
        probabilities_given_scottish.append(
            (len(list(filter(lambda x: x[i] == 1, scottish))) + 1) / (len(scottish) + k))
        probabilities_given_english.append((len(list(filter(lambda x: x[i] == 1, english))) + 1) / (len(english) + k))

    probScottish = len(scottish) / len(people)
    probEnglish = len(english) / len(people)
    return probabilities_given_english, probabilities_given_scottish,probEnglish, probScottish


def classify_input(qualities):
    probabilities_given_english, probabilities_given_scottish, probability_english, probability_scottish = classifier()
    for idx, i in enumerate(qualities):
        if i == 1:
            probability_english = probability_english * (probabilities_given_english[idx])
            probability_scottish = probability_scottish * (probabilities_given_scottish[idx])
        else:
            probability_english = probability_english * (1 - probabilities_given_english[idx])
            probability_scottish = probability_scottish * (1 - probabilities_given_scottish[idx])
    total_prob = (probability_scottish + probability_english)
    return probability_scottish/total_prob, probability_english/total_prob


if __name__ == "__main__":
    print(classify_input([1, 0, 1, 1, 0]))
    print(classify_input([0, 1, 1, 0, 1]))
