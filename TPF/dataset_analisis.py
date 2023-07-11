import matplotlib.pyplot as plt


def univariable_analisis(df):
    for column in df.columns:
        plt.figure()
        plt.boxplot(df[column])
        plt.title(column)
        plt.savefig('./images/' + column)
        plt.show()

    plt.scatter(df['TotalMinutesAsleep'], df['Calories'])
    plt.xlabel('Total Minutes Asleep')
    plt.ylabel('Calories')
    plt.title('Calories vs Total Minutes Asleep')
    plt.savefig('./images/calories_vs_total_minutes_asleep')
    plt.show()
