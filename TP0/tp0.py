import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def replaceMissingDataMedian(name):
    f = filter(lambda x: x!=999.99,data[name] )
    l= list(f)
    me = np.median(l)
    mapMedian = map(lambda x: x if x!=999.99 else me,data[name])
    return mapMedian
   

def replaceMissingDataAvarage(name):
    f = filter(lambda x: x!=999.99,data[name] )
    l= list(f)
    avg = np.average(l)
    mapAvg = map(lambda x: x if x!=999.99 else avg,data[name])
    return mapAvg


def plotAverageMedianCompare(name):
    mapMedian = replaceMissingDataMedian(name)
    mapAvg = replaceMissingDataAvarage(name)
    d = {'avg': np.array(list(mapAvg)), 'median': np.array(list(mapMedian))}
    fig, ax = plt.subplots()
    ax.boxplot(d.values())
    ax.set_xticklabels(d.keys())
    ax.set_title(name)


    plt.show()
    

def covarianceSex(name1):
    map1 = replaceMissingDataAvarage(name1)
    mapSex = map(lambda x: 0 if x=="F" else 1,data["Sexo"])
    df = pd.DataFrame({'name': np.array(list(map1)), 'sex': np.array(list(mapSex))})
    sns.heatmap(df.corr(), annot=True)
    sns.set_title(name1)
    plt.show()
    

def dispersionSex(name1):
    map1 = replaceMissingDataAvarage(name1)
    mapSex = map(lambda x: 0 if x=="F" else 1,data["Sexo"])
    y = np.array(list(mapSex))
    x = np.array(list(map1))
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    ax.set_title(name1)

    
    plt.show()
    
    


if __name__ == "__main__":
    data = pd.read_excel('./datosTrabajo.xls')
    #plotAverageMedianCompare("Alcohol")
    #plotAverageMedianCompare("Grasas_sat")
    plotAverageMedianCompare("Calorías")
    #dispersionSex("Calorías")
    

