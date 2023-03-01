import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Read the excel file



def plotAverage(name):
    f = filter(lambda x: x!=999.99,data[name] )
    l= list(f)

    avg = np.average(l)
    me = np.median(l)

    print(avg)
    print(me)

    mapMedian = map(lambda x: x if x!=999.99 else me,data[name])
    mapAvg = map(lambda x: x if x!=999.99 else avg,data[name])
    
    
    d = {'avg': np.array(list(mapAvg)), 'median': np.array(list(mapMedian))}
    fig, ax = plt.subplots()
    ax.boxplot(d.values())
    ax.set_xticklabels(d.keys())
    ax.set_title(name)


    plt.show()
    


if __name__ == "__main__":
    data = pd.read_excel('./datosTrabajo.xls')

    plotAverage("Alcohol")
    #plotAverage("Grasas_sat")
    #plotAverage("Calorías")
    

