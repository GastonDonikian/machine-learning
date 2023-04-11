import pandas as pd

# class Student:
#     def __init__(self, admit, gre, gpa, rank):
#         self.admit = admit == 1
#         self.gre = gre >= 500
#         self.gpa = gpa >= 3.0
#         self.rank = rank
#         for 

ranks = [1,2,3,4]
bool_opt = [True, False]
admit_opt = [0,1]

def classifier():
    data = pd.read_csv('./resources/binary.csv')
    #students = []

    data.gre[data.gre < 500] = False
    data.gre[data.gre >= 500] = True
    data.gpa[data.gpa < 3] = False
    data.gpa[data.gpa >= 3] = True
    
    total_rows = data.shape[0]
    


    probs_admitted_gpa = {}
    probs_admitted_gre = {}
    probs_gpa = {}
    probs_gre = {}

    probs_gre, probs_gpa, probs_admitted_gpa, probs_admitted_gre = create_maps()
    probs_rank = []

    #pd.options.mode.chained_assignment = None

 
    for rank in ranks:
        probs_rank.append(data[data['rank'] == rank].shape[0] / total_rows)
        rows_rank = data[data['rank'] == rank].shape[0]
        for gre in bool_opt:   
            probs_gre[rank][gre] = (data.query("rank == " + str(rank) + " and gre == "+ str(gre)).shape[0]+1)/(rows_rank+2)
            rows_gre_rank = data.query("rank == " + str(rank) + " and gre == "+ str(gre)).shape[0]
            for admit in admit_opt:
                probs_admitted_gre[rank][gre][admit] = (data.query("rank == " + str(rank) + " and gre == "+str(gre)+ " and admit == " + str(admit)).shape[0] + 1)/(rows_gre_rank + 2)
        for gpa in bool_opt:
            rows_gpa_rank = data.query("rank == " + str(rank) + " and gpa == "+ str(gpa)).shape[0]
            probs_gpa[rank][gpa] = (data.query("rank == " + str(rank) + " and gpa == "+ str(gpa)).shape[0] + 1)/(rows_rank + 2)
            for admit in admit_opt:
                probs_admitted_gpa[rank][gpa][admit] = (data.query("rank == " + str(rank) +  " and gpa == "+str(gpa)+" and admit == " + str(admit)).shape[0] + 1)/ (rows_gpa_rank +2)

    print(probs_admitted_gpa)
    print("-----------")
    print(probs_admitted_gre)
    print("-----------")
    print(probs_gpa)
    print("-----------")
    print(probs_gre)
    print("-----------")
    print(probs_rank)
    print("-----------")

    # print("Prob rank 1 and admitted")
    # nnn = data.query("rank == 1 and admit == 0").shape[0]/total_rows
    # print(nnn)
    # print("Prob rank 1 and admitted dividido por prob rank 1")
    # print(nnn/(data.query("rank == 1").shape[0]/total_rows))


    return probs_rank,probs_gre, probs_gpa, probs_admitted_gpa, probs_admitted_gre

    # a)
    # Para chequear








def a(probs_rank, probs_gre, probs_admitted_gre):
    result = 0
    for gre in bool_opt:
            result += probs_admitted_gre[1][gre][0]*probs_gre[1][gre]*probs_rank[0]
    
    print(result)
    result = result/probs_rank[0]
    print(result)
    #data.query("rank == 1 and admit == 1").shape[0]/total_rows

def b(probs_rank, probs_gre, probs_gpa, probs_admitted):
   print("gola") 
    

def create_maps(): 
  
    probs_admitted_gpa = {}
    probs_admitted_gre = {}
    probs_gpa = {}
    probs_gre = {}
   
    for rank in ranks:
        probs_gre[rank] = {}
        probs_gpa[rank] = {}
        probs_admitted_gpa[rank] = {}
        probs_admitted_gre[rank] = {}
        for gre in bool_opt:
            probs_gre[rank][gre] = {}
            probs_admitted_gre[rank][gre] = {}
            for gpa in bool_opt:
                probs_admitted_gpa[rank][gpa]={}
                probs_gpa[rank][gpa] = {}
                for admit in admit_opt:
                    probs_admitted_gre[rank][gre][admit]={}
                    
                    probs_admitted_gpa[rank][gpa][admit]={}

    return probs_gre, probs_gpa, probs_admitted_gpa, probs_admitted_gre



    
if __name__ == "__main__":
    probs_rank, probs_gre, probs_gpa, probs_admitted_gpa, probs_admitted_gre = classifier() 
    a(probs_rank, probs_gre, probs_admitted_gre)