import re
import random
import standardalize
from numpy import transpose


def range_custom(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1


total_matrix = [[]]
feature_list_of_all_instances = []
class_list_of_all_instances = []
total_matrix = []
Total_data_number = int(291920)
# Total_data_number = 10000
# clf = MLPClassifier(hidden_layer_sizes=5000,activation='logistic',learning_rate='adaptive')
data = []  # this list is to generate index value for k fold validation

print("Opening  Text ...  ")

file_read = open('/home/farshid/Desktop/nr_dataset_1294.txt', 'r')
count_for_number_of_instances = 0
i = 0

# for i in range(0, 1287):
#     file_read.readline()
print("Opening  Text ...  ")
count_for_number_of_instances = 0
i = 0
cout = 0
print("Starting To read From Text ...  ")
with open('/home/farshid/Desktop/nr_dataset_1477.txt', 'r') as file_read:
    for x in file_read:
        if len(x) <= 10:
            break
        l = x.rstrip('\n').split(',')
        l = list(map(float, l))
        total_matrix.append(l)
        # feature_list_of_all_instances.append(l[0:519])
        # class_list_of_all_instances.append(int(l[519]))
        i += 1
        #
        # if i == Total_data_number:
        #     break

c = 0

print("Total instances ", len(total_matrix))
print("Total Features ", len(total_matrix[0]) -1)


print("Starting To Standardize Total Matrix ...  ")
# print(total_matrix[0][882:])
total_matrix = standardalize.std(total_matrix, 882, 522 + 73)
# print(total_matrix[0][882:])
print("Total instances ", len(total_matrix))
print("Total Features ", len(total_matrix[0])-1)

for l in total_matrix:
    index = len(l) -1
    # print(index)
    feature_list_of_all_instances.append(l[0:index])
    class_list_of_all_instances.append(l[index])

for i in class_list_of_all_instances:
    if i == 1:
        c += 1
print("Positive data  ", c)

count = 0
redunrent_list = []
for i in range(0,1477):
    # print("checking ", i,"th column" )
    if ( all(value[i] ==0 for value in feature_list_of_all_instances) ):
        # print(i,"th feature redundant")
        count += 1
        redunrent_list.append(i)
        continue
    if (all(value[i] == 1 for value in feature_list_of_all_instances)):
        # print(i, "th feature redundant")
        count += 1
        redunrent_list.append(i)
        continue

count2 = 0


print("count for redundant features in 881 matrix  " , count)
# print(redunrent_list)

# count = 0
# count2 = 0
# relevent_list = []
#
# for i in range(882,1294):
#     count = 0
#     for j in range(0,Total_data_number):
#         if class_list_of_all_instances[j] == feature_list_of_all_instances[j][i]:
#             count += 1
#     value = count/Total_data_number*100
#     if value > 70:
#         # print(i,"th matched " , value ,"% with class data")
#         count2 += 1
#         relevent_list.append(i)
#     if value < 30:
#         # print(i, "th matched ", value, "% with class data")
#         count2 += 1
#         relevent_list.append(i)
#
# print("approx relevant features  ",count2)
# # print(redunrent_list)
# # print(relevent_list)
# final_list = []
#
# for i in relevent_list:
#     if i not in redunrent_list:
#         final_list.append(i)
#
#
# # print(final_list)
# print("count length of final list " , len(final_list))
#
# feature_matrix = transpose(feature_list_of_all_instances)
#
#
# reduced_feature_matrix = []
# for i in range(0, 1294):
#     if i not in redunrent_list:
#         reduced_feature_matrix.append(feature_matrix[i])
#
#
# feature_matrix = reduced_feature_matrix.tolist()
#
# feature_list_of_all_instances = feature_matrix
# print("total instances", len(feature_list_of_all_instances))
# print("total features " ,len(feature_list_of_all_instances[0]))
