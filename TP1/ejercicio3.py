import pandas as pd

# class Student:
#     def __init__(self, admit, gre, gpa, rank):
#         self.admit = admit == 1
#         self.gre = gre >= 500
#         self.gpa = gpa >= 3.0
#         self.rank = rank
#         for

ranks = [1, 2, 3, 4]
bool_opt = [True, False]
admit_opt = [0, 1]


def classifier():
    data = pd.read_csv('./resources/binary.csv')

    data.gre[data.gre < 500] = False
    data.gre[data.gre >= 500] = True
    data.gpa[data.gpa < 3] = False
    data.gpa[data.gpa >= 3] = True

    total_rows = data.shape[0]

    probs_admitted = {}
    probs_gpa = {}
    probs_gre = {}
    probs_rank = {}

    for rank in ranks:
        rows_rank = data[data['rank'] == rank]
        probs_rank[rank] = (rows_rank.shape[0] + 1) / (total_rows + 4)
        probs_gre[rank] = {}
        probs_gpa[rank] = {}
        probs_admitted[rank] = {}
        for gre in bool_opt:
            probs_gre[rank][gre] = (rows_rank.query(
                "gre == " + str(gre)).shape[0]+1)/(rows_rank.shape[0]+2)
        for gpa in bool_opt:
            probs_gpa[rank][gpa] = (rows_rank.query(
                " gpa == " + str(gpa)).shape[0] + 1)/(rows_rank.shape[0] + 2)

        for gre in bool_opt:
            probs_admitted[rank][gre] = {}
            for gpa in bool_opt:
                probs_admitted[rank][gre][gpa] = {}
                for admit in admit_opt:
                    rows_rank_gre_gpa = data.query(
                        "rank ==" + str(rank) + " and gre == "+str(gre) + " and gpa == " + str(gpa))
                    probs_admitted[rank][gre][gpa][admit] = (rows_rank_gre_gpa.query(
                        "admit == " + str(admit)).shape[0] + 1) / (rows_rank_gre_gpa.shape[0] + 2)

    print("-----------")
    print(probs_gpa)
    print("-----------")
    print(probs_gre)
    print("-----------")
    print(probs_rank)
    print("-----------")

    return probs_rank, probs_gre, probs_gpa, probs_admitted

    # a)
    # Para chequear


def a(probs_rank, probs_gre, probs_gpa, probs_admitted):
    numerator = 0
    denominator = 0
    rank = 1
    admit = 0

    for gre in bool_opt:
        for gpa in bool_opt:
            numerator += probs_admitted[rank][gre][gpa][admit] *\
                probs_gre[rank][gre] *\
                probs_gpa[rank][gpa] *\
                probs_rank[rank]
            denominator += probs_gre[rank][gre] *\
                probs_gpa[rank][gpa] *\
                probs_rank[rank]

    print(numerator/denominator)


def b( probs_admitted):
    rank = 2
    admit = 1
    gre = False
    gpa = True

    result = probs_admitted[rank][gre][gpa][admit]

    print(result)


if __name__ == "__main__":
    probs_rank, probs_gre, probs_gpa, probs_admitted = classifier()
    a(probs_rank, probs_gre, probs_gpa, probs_admitted)
    b(probs_admitted)
