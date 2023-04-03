import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def replace_missing_data_median(name):
    f = filter(lambda x: x != 999.99, data[name])
    l = list(f)
    me = np.median(l)
    print("La media de " + name + " es: " + str(me))
    map_median = map(lambda x: x if x != 999.99 else me, data[name])
    return map_median


def replace_missing_data_average(name):
    f = filter(lambda x: x != 999.99, data[name])
    l = list(f)
    avg = np.average(l)
    print("El promedio de " + name + " es: " + str(avg))
    map_avg = map(lambda x: x if x != 999.99 else avg, data[name])
    return map_avg


def plot_average_median_compare(name):
    map_median = replace_missing_data_median(name)
    map_avg = replace_missing_data_average(name)
    d = {'avg': np.array(list(map_avg)), 'median': np.array(list(map_median))}
    fig, ax = plt.subplots()
    ax.boxplot(d.values())
    ax.set_xticklabels(d.keys())
    ax.set_title(name)

    plt.show()


def covariance_sex(name1):
    map1 = replace_missing_data_average(name1)
    map_sex = map(lambda x: 0 if x == "F" else 1, data["Sexo"])
    df = pd.DataFrame({'name': np.array(list(map1)), 'sex': np.array(list(map_sex))})
    sns.heatmap(df.corr(), annot=True)
    sns.set_title(name1)
    plt.show()


def dispersion_sex(name1):
    map1 = replace_missing_data_average(name1)
    map_sex = map(lambda x: 0 if x == "F" else 1, data["Sexo"])
    y = np.array(list(map_sex))
    x = np.array(list(map1))
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(name1)

    plt.show()


def get_index(name) -> int:
    return {
        "Grasas_sat": 0,
        "Alcohol": 1,
        "Calorías": 2
    }[name]


def create_biplot(x_axis: str, y_axis: str):
    x_index = get_index(x_axis)
    y_index = get_index(y_axis)
    clean_data = list(filter(lambda x: x[x_index] != 999.99 and
                                       x[y_index] != 999.99, data.values.tolist()))
    # plt.scatter([elem[x_index] for elem in clean_data],
    #             [elem[y_index] for elem in clean_data])
    #
    index_data_male = [elem[x_index] for elem in clean_data if elem[3] == 'M']
    plt.scatter(index_data_male, range(len(index_data_male)), color='blue')
    # plt.scatter([elem[x_index] for elem in clean_data if elem[3] == 'F'],
    #             [elem[y_index] for elem in clean_data if elem[3] == 'F'], color='pink')

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(x_axis + " vs. " + y_axis)
    # plt.vlines(x=[1100, 1700], ls='--', lw=1,ymin=0, ymax=40)
    plt.show()


if __name__ == "__main__":
    data = pd.read_excel('./datosTrabajo.xls')
    # replace_missing_data_average("Alcohol")
    # replace_missing_data_average("Grasas_sat")
    # replace_missing_data_average("Calorías")
    # replace_missing_data_median("Alcohol")
    # replace_missing_data_median("Grasas_sat")
    # replace_missing_data_median("Calorías")
    # create_biplot("Calorías", "Alcohol")
    # create_biplot("Alcohol", "Calorías")
    # create_biplot("Calorías", "Grasas_sat")
    # plot_average_median_compare("Alcohol")
    # plotAverageMedianCompare("Grasas_sat")
    # plotAverageMedianCompare("Calorías")
    # dispersion_sex("Calorías")