import re

import numpy
import numpy as np
import mysql.connector

# file.readline()


# print("legit         ......................")
# print(file.seek(1,1) )



#

hsa_list = []
hsa_list_temp = []
cnx = mysql.connector.connect(user='root', password='123',
                              host='127.0.0.1',
                              database='ir_dataset')
cursor = cnx.cursor()

query = ("SELECT hsa FROM hsaSeq")
x = cursor.execute(query)

for i in cursor:
    # print(i)

    hsa_list.append(i)

cnx.close()

# print(hsa_list[2][0])
count = 0
# for l in range(0,len(hsa_list)):
#
#     hsa_list[l] = hsa_list[l][0]
#     x = hsa_list[l][0:3]
#     y = hsa_list[l][4:]
#
#     hsa_list[l] = x  + y

map = {}

for i in hsa_list:
    # print(i)
    # i = str(i)
    i = ''.join(i)
    x = i[0:3]
    y = i[4:]
    i = x + y + '.txt.spd3'
    j = x + y

    # hsa_list.append(i)
    hsa_list_temp.append(i)

# print(" g")

hsa_count = 0

for hsa in hsa_list_temp:
    # z = hsa_list.index(i)
    # hsa  = hsa_list_temp[z]
    print(hsa)
    file_read = open('/home/farshid/Desktop/IR_seqs/' + hsa, 'r')
    file = file_read.readline()

    # file = file_read.readline()
    # print(file)


    final_list = []

    num_lines = sum(1 for line in open('/home/farshid/Desktop/IR_seqs/' + hsa, 'r'))
    # num_lines -= 1
    matrix = [[0 for x in range(8)] for y in range(8)]
    probability_matrix = [[0 for x in range(3)] for y in range(3)]

    i = 0

    a = []

    while i < num_lines - 1:
        i += 1

        str1 = file_read.readline()
        # print(str1)
        # str1 = re.findall("-?\d+", str1)

        l = str1.rstrip('\n').split('\t')
        # l = list(l)
        # l = list(map(str, l[3:10]))
        # str1 = str1[11:]
        # print(l[3:])
        # l = ','.join(l[3:])
        # print(l)
        a.append(l[3:])

    array = a
    # print(a)
    # break

    degree_matrix = [[0 for x in range(8)] for y in range(num_lines-1)]

    degree_index = 0

    for x in range(0,num_lines-1):
        degree_index = 0
        for y in range(1,5):
            # print("x = ", x , " y = ", y , " deg_index = ",degree_index)
            degree_matrix[x][degree_index] = numpy.math.sin(float(array[x][y]) * numpy.pi / 180 )
            degree_matrix[x][degree_index+1] = numpy.math.cos(float(array[x][y]) * numpy.pi / 180  )
            degree_index += 2

        # print()


    # print(len(array))
    # print(len(array[0]))
    # print(len(degree_matrix))
    # print(len(degree_matrix[0]))

    temp_array = array

    array = degree_matrix

    # print(num_lines - 1)

    # print(array[0])
    # print()
    # print(degree_matrix[0])
    #
    #

    #
    #

    num_lines -= 1

    for k in range(0, 8):
        for l in range(0, 8):
            for i in range(0, num_lines - 1 ):
                # print("k = ", k, " l = ", l, " i = ", i)
                # r = float(array[i][k]) * float(array[i + 1][l])
                # print(r)
                matrix[k][l] += float(array[i][k]) * float(array[i + 1][l])
            matrix[k][l] = matrix[k][l] / (num_lines - 1)

    # print(matrix)
    # print(" rows ", len(matrix), " cols ", len(matrix[0]))

    array = temp_array

    for k in range(0, 3):
        for l in range(5, 8):
            for i in range(0, num_lines - 1 ):
                # print("k = ", k, " l = ", l, " i = ", i)
                probability_matrix[k][l- 5] += numpy.float(array[i][k]) * numpy.float(array[i + 1][l -5 ])
                # print(probability_matrix[k][j-5])
            probability_matrix[k][l - 5] = probability_matrix[k][l - 5] / (num_lines - 1)
            # print(k,i,j)

    # print(probability_matrix)
    # print(" rows ", len(probability_matrix), " cols ", len(probability_matrix[0]))




    final_matrix = [[0 for x in range(73)] for y in range(1)]
    index = 0
    for m in range(0, len(matrix)):
        for n in range(0, len(matrix[m])):
                final_matrix[0][index] = matrix[m][n]
                index += 1
    # index = 0
    for m in range(0, len(probability_matrix)):
        for n in range(0, len(probability_matrix[0])):
                final_matrix[0][index] = probability_matrix[m][n]
                index += 1


    # print(final_matrix[0])
    # print(" rows ", len(final_matrix), " cols ", len(final_matrix[0]))

    # break

    matrix = final_matrix
    # print(len(matrix[0]))
    # print(len(matrix))
    # print(matrix[0])

    # print()
    # print()

    # file = open('/home/farshid/Desktop/IR_seqs/' + hsa + '_bigram.txt', 'w')
    #
    # for m in range(0, len(matrix)):
    #     for n in range(0, len(matrix[m]) ):
    #
    #         if n == len(matrix[m])-1 and m == len(matrix) - 1:
    #             file.write(str(matrix[m][n]) + "\n")
    #         else:
    #             file.write(str(matrix[m][n]) + ",")
    #
    # file.close()
    # break








    # break
# #
#     hsa_count += 1

    # if hsa_count == 2:
    #     break
    # break









    #
    #
    # file.write(str(m) +str(n) + ",")
    #
    #
    #
    # file.close()  # print( int ( do_stuff.array[0][2] ) * 2 )
    # count+=1
    #
    #
    # val = float(count/len(hsa_list))*100
    # val = int(val)
    # val = str(val)
    # print(val+"% completed")
    # #

    # break  # matrix



#
# file.readline()
# str2 = file.read(89)
# str2 = str2[11:]
# b=[]
#
#
#
# for i in str2:
#      if (  ( i >= '0' and i <= '9' ) or i ==',' or i == '-' ):
#         b.append(i)
# print(b)
#
#
#
# l = []
# l.append('-5')
# print(l)
#
#
# # file.readline()
# # str2 = file.read(89)
# # str2 = str2[11:]
# # b = []
# # b += str2
# # print(b)
#
# # print(a*b)
