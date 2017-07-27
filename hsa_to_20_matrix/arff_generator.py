import re
import numpy as np
import arff
import mysql.connector
from sklearn import svm


# data = arff.load(open('wheater.arff', 'rb'))



# arff_writer.pytypes[arff.nominal] = '{angry,disgusted,fearful,happy,sad,surprised}'
# arff_writer.write([arff.nominal('emotion')])
# file.readline()


# print("legit         ......................")
# print(file.seek(1,1) )




# class do_stuff:

#######################################################################################
## make the 881 sized matrix form 0 - 881 . there is chance it has a 882 Size . the result
## are stored in drug_list dictionary


cnx = mysql.connector.connect(user='root', password='123',
                              host='127.0.0.1',
                              database='thesisDataset')
cursor = cnx.cursor()

query = ("select data.DrugId , smileyWithFingerprint.compound from smileyWithFingerprint,data where data.Compound = smileyWithFingerprint.smiley")
x = cursor.execute(query)

drug_list = {}

def list_filler(list):
    return_this_list = [0]*882

    for i in list:
        # print(i)
        return_this_list[int(i)] = 1

    return return_this_list

counter = 0
for i in cursor:
    # print(i[0])
    rr = len(i[0])
    str1 = i[1]
    r = len(str1)
    str1 = re.findall("-?\d+", str1)
    r = len(str1)
    # my_list = str1.split(",")
    # print(len(str1))
    drug_list[i[0]] = list_filler(str1)
    # hsa_list.append(i)


# for k,v in drug_list.items():
#     print(k, v)

#########################################################################################
#########################################################################################



cnx = mysql.connector.connect(user='root', password='123',
                              host='127.0.0.1',
                              database='thesisDataset')
cursor = cnx.cursor()

query = ("select distinct(Protein) from Connection")
x = cursor.execute(query)

hsa_dict = {}
for i in cursor:

    x = i[0][0:3]
    y = i[0][3:]
    hsa = x + ':' + y
    # print(hsa)
    file_read = open('/home/farshid/PycharmProjects/Experiments_for_SS/hsa_to_20_matrix/x/' + hsa + '.txt', 'r')
    x = file_read.readline()
    l = re.findall("-?\d+", x)
    hsa_dict[hsa] = l
    # str1 = i[1]
    # str1 = re.findall("-?\d+", str1)

    # drug_list[i[0]] = list_filler(str1)

cnx.close()

# for k,v in hsa_dict.items():
#     print(k, v)

# print(len( drug_list['D00002']) )
# print(len(hsa_dict['hsa:10']) )


###############################################################################################################
############################################################################################################
print("Writing ...")
file = open('/home/farshid/Desktop/dataset_1282.txt', 'w')
# file.write(write_this_str)

#### arff file creation pattern

# file.write("@relation Drug_target\n")
# file.write("\n")
# file.write("\n")
# i = 0
# for i in range(0,1282):
#     temp_str = "@ATTRIBUTE "+str(i)+"    NUMERIC\n"
#     file.write(temp_str)
#
# temp_str = "@ATTRIBUTE class    {0,1}\n"
#
# file.write(temp_str)
#
# file.write("\n")
# file.write("\n")
# file.write("\n")
#
# file.write("@DATA")
# file.write("\n")


#### arff file creation pattern
#
# file.write("0,0,1\n")
# file.write("0,1,0\n")
# file.write("1,0,1\n")
# file.write("1,1,0\n")

cnx = mysql.connector.connect(user='root', password='123',
                              host='127.0.0.1',
                              database='thesisDataset')
cursor = cnx.cursor()

query = ("select Connection.DrugId,Connection.Protein,Connection.Relation from Connection limit 300000")
x = cursor.execute(query)
counter = 0
flag_complete = 0
feature_list = []
class_list = []

class_matrix = []

count = 0

for i in cursor:

    # print(i[0])
    x = i[1][0:3]
    y = i[1][3:]
    hsa = x + ':' + y
    # # print(hsa)
    # # print(i[2])
    #
    #
    # print(len(drug_list[i[0]]))
    # print(len(hsa_dict[hsa]))
    # print()
    str1  = ','.join(str(i) for i in drug_list[i[0]])
    # str1 = ''
    str2  = ','.join(str(i) for i in hsa_dict[hsa])
    # print(len(str1))
    # print(len(str2))
    # print()
    str3 = i[2]

    write_this_str = str1 + "," + str2 + "," + str(str3) + "\n"




    #
    # if counter == 10:
    #     break
    # counter = counter + 1

    my_list = write_this_str.split(",")
    my_list = list(map(int , my_list))


    file.write(write_this_str)
    # class_matrix.append(write_this_str)
    count += 1

    completed = int( count / 291920 * 100)
    if completed > flag_complete:
        flag_complete = completed
        print("Completed " , flag_complete , "% ")

    # print(len(my_list) )
    # print(my_list[1281])
    # feature_list.append(my_list[0:1282])
    # class_list.append(my_list[1282])
    # print(len(feature_list[0]))
    # print(len(class_list[0]))

cnx.close()



#
clf = svm.SVC()
#

# clf.fit(feature_list,class_list)


# print(len(feature_list[0]))

# str = class_list[0]
# my_list = str.split(",")
# print(len(my_list))


# print(my_list)

# print(len(class_list[0]))

# for i in range (0,10):
#     print(clf.predict([feature_list[i]]) )
#



