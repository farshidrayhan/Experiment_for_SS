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
                              database='gpcr_dataset')
cursor = cnx.cursor()

query = (
"select smiley_fingerprint.drugid,smiley_fingerprint.fingerprint from smiley_fingerprint")
x = cursor.execute(query)

drug_list = {}


def list_filler(list):
    return_this_list = [0] * 882

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
    # print(len(drug_list[i[0] ]))

    # hsa_list.append(i)

# for k,v in drug_list.items():
#     print(k, v)

#########################################################################################
#########################################################################################



cnx = mysql.connector.connect(user='root', password='123',
                              host='127.0.0.1',
                              database='gpcr_dataset')
cursor = cnx.cursor()

query = ("select distinct(Protien) from connection")
x = cursor.execute(query)

hsa_dict = {}
for i in cursor:
    x = i[0][0:3]
    y = i[0][4:]
    hsa = x + y
    # print(hsa)
    file_read = open('/home/farshid/Desktop/NR_seqs/' + hsa + '.txt', 'r')
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
cnx = mysql.connector.connect(user='root', password='123',
                              host='127.0.0.1',
                              database='gpcr_dataset')
cursor = cnx.cursor()

query = ("select hsa,data from hsa_structure_2")
x = cursor.execute(query)

structure_dict = {}
for i in cursor:
    structure_dict[i[0]] = i[1]

#############################################################################################################
##########################################################
cnx.close()###################################################
file = open('/home/farshid/Desktop/gpcr_dataset_1282.txt', 'w')
# file.write(write_this_str)

#### arff file creation pattern
#
# file.write("@relation Drug_target_nr\n")
# file.write("\n")
# file.write("\n")
# i = 0
# for i in range(0,1294):
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




# exclude_list = ['D00187','D00443','D00188','D00348','D00129','D00299','D00930','D00211','D00246','D01387']

# cnx = mysql.connector.connect(user='root', password='123',
#                               host='127.0.0.1',
#                               database='nr_dataset')
# query = "select distinct(connection.DrugId) from data,connection where data.DrugId = connection.DrugId "
# cursor = cnx.cursor()
# x = cursor.execute(query)
# for i in cursor:
#     i = str(i)
#     i = i[2:8]
#
#     exclude_list.append(i)
#
# cnx.close

cnx = mysql.connector.connect(user='root', password='123',
                              host='127.0.0.1',
                              database='nr_dataset')
cursor = cnx.cursor()

query = ("select connection.DrugId,connection.Protien,connection.Relation from connection limit 300000")
x = cursor.execute(query)
counter = 0

feature_list = []
class_list = []

class_matrix = []
print("Concating   ")
for i in cursor:

    # print(i[0])
    x = i[1][0:3]
    y = i[1][4:]
    hsa = x + y
    # # print(hsa)
    # # print(i[2])
    #
    #
    # print(len(drug_list[i[0]]))
    # print(len(hsa_dict[hsa]))
    # list = ['D00950','D00961']

    # if  exclude_list.__contains__(i[0]):
    #     print( "skipped " , counter)
    #     counter+=1
    #     continue
    # print()
    # if

    str1 = ','.join(str(i) for i in drug_list[i[0]])
    # str1 = ''
    str2 = ','.join(str(i) for i in hsa_dict[hsa])

    str3 = ''.join(str(i) for i in structure_dict[hsa])
    # print(len(str1))
    # print(len(str2))
    # print()
    str4 = int(i[2])
    if int(i[2]) ==  1:
        counter += 1
    # print(str(str4))

    write_this_str = str1 + "," + str2 + "," + str3 + ',' + str(str4) + "   \n"
    # write_this_str = str1 + "," + str2  + ',' + str(str4) + "   \n"

    # rt = list(map(float, write_this_str))
    # print(len(wr))
    file.write(write_this_str)
    # break
    # #
    # if counter == 100:
    #     break
    # counter = counter + 1

    # my_list = write_this_str.split(",")
    # my_list = list(map(int , my_list))

file.close()
# class_matrix.append(write_this_str)
print("counter for cls 1  ", counter)
# print(len(my_list) )
# print(my_list[1281])
# feature_list.append(my_list[0:1282])
# class_list.append(my_list[1282])
# print(len(feature_list[0]))
# print(len(class_list[0]))

cnx.close()



#
# clf = svm.SVC()
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
