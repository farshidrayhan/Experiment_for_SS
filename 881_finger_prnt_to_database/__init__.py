import re
import numpy as np
import mysql.connector

# class do_stuff:
# hsa = 'hsa:10269.seq'
# path = '/home/farshid/PycharmProjects/Experiment_for_SS/hsa_to_20_matrix/' + hsa + '.hsa_to_20_matrix'
# file = open(path, 'r')
# file.readline()
# file.readline()
# final_list = []
# num_lines = sum(1 for line in open(path, 'r'))
# num_lines = num_lines - 6 - 3  # 6 from the bottom 3 from the top
# array = np.zeros(shape=(num_lines, 20))
# hsa_list = []

# matrix = [[0 for x in range(20)] for y in range(20)]

# def __init__(self):

smiley_list = []
fingerprint_list = []
cnx = mysql.connector.connect(user='root', password='123',
                              host='127.0.0.1',
                              database='thesisDataset')
cursor = cnx.cursor()

query = ("SELECT smiley,fingerprint FROM smiley_fingerprint")
x = cursor.execute(query)

for i in cursor:
    smiley_list.append(i[0])
    fingerprint_list.append(i[1])

cnx.close()
# print(fingerprint_list[0])

# str1 = "10 12 13"
# str1 = re.findall("\d+", str1)
# print(str1)


for j in range(0, 426):
    x = re.findall("\d+", fingerprint_list[j])
    x = list(map(int, x))
    # print(x)
    fingerprint_list[j] = x

# fingerprint_list
# for j in range(0, 426):
#     fingerprint_list[j] = [int(i) for i in fingerprint_list[j]]
#
final_list = []

list = [0] * 881
for j in range(0, 426):
    # print()
    # print(j)
    var = fingerprint_list[j]
    # print(var)

    for i in range(0, len(var)):
        x = var[i]
        # print( str( x ) + " " , end='')
        list[x] = 1
    final_list.append(list)

    # print()
    # print(list)
    list = [0] * 881

hsa_list = []
cnx = mysql.connector.connect(user='root', password='123',
                              host='127.0.0.1',
                              database='nr_Dataset')
cursor = cnx.cursor()

query = ("SELECT hsa FROM hsaSeq")
x = cursor.execute(query)

for i in cursor:
    # print(i)
    hsa_list.append(i)

cnx.close()
count = 0
final_list_multiplied_values = []
for l in range(0, len(hsa_list)):
    hsa_list[l] = hsa_list[l][0]
    x = hsa_list[l][0:3]
    y = hsa_list[l][3:]

    hsa_list[l] = x + ':' + y

for hsa in hsa_list:
    path = '/home/farshid/PycharmProjects/Experiments_for_SS/hsa_to_20_matrix/x/' + hsa + '.txt'
    file = open(path, 'r')
    str = re.findall("\d+", file.readline())

    final_list_multiplied_values.append(str)
    # print(hsa)
    # print(str)

# print(hsa_list[2][0])
file_write = open("/home/farshid/Desktop/data.arff", "w")

vr = final_list_multiplied_values[0]
str1 = ""
str2 = ""

for i in range(0, 400):
    str1 += vr[i] + ","


print(str1)
# str = str1 + " ," + str2
# file_write.write(str)

y = [1, 2]
# z = [3,4]
# x.append(y)
# x.append(z)
# print(x)
# print(fingerprint_list[0])  # print(fingerprint_list[0])





# str1 = "CREATE TABLE `thesisDataset`.`hsaWithFingerprint` ( \n"
# str2 =   "  `hsa` VARCHAR(3000) NOT NULL,\n"
# str3 = ""
# for j in range(0, 426):
#     str3 += "   `"+str(j)+"`BLOB NULL,\n"
#
# str4 = "  PRIMARY KEY (`hsa`));"
#
# str = str1+str2+str3+str4
#
# print(str)



# def do_shit(self):
#
#         str = self.file.read(89)
#         str = str[11:]
#         a = []
#
#         str = re.findall("-?\d+", str)
#
#         self.file.readline()
#
#         return str
#
#     def create_final_list(self):
#
#         self.do_shit()
#         i = 0
#
#         while i < do_stuff.num_lines:
#             do_stuff.final_list.append(self.do_shit())
#             i = i + 1
#
#         do_stuff.array = do_stuff.final_list
#
#     def get_matrix(self):
#         for m in range(0, 20):
#             for n in range(0, 20):
#                 for i in range(0, self.num_lines - 1):
#                     self.matrix[m][n] += int(self.array[i][m]) * int(self.array[i + 1][n])
#
#     def write(self, matrix):
#
#         file = open(self.hsa + '.csv', 'w')
#         for m in range(0, len(matrix)):
#             for n in range(0, len(matrix[m])):
#                 file.write(str(matrix[m][n]) + ", ")
#             file.write("\n")
#         file.close()
#
#
# do = do_stuff()
#
# do.create_final_list()
# do.get_matrix()
# do.write(do.matrix)
#
