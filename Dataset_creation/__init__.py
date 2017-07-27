from collections import Counter
i = 0
hsa_matrix = []
drug_matrix = []
Total_data_number = 0
line = 0
with open('bind_orfhsa_drug_e.txt', 'r') as file_read:
    for x in file_read:
        if len(x) <= 10:
            break
        l = x.rstrip('\n').split('	')
        # l = re.findall("\d+\.\d+|-?0\.\d+|-?\d+", l)
        # print(l)
        # l = list(map(float, l))
        # l = list(map(int, l))
        # print(l[519])
        # if l[519] == 1:
        #     cout += 1
        #     print(cout)
        hsa_matrix.append(l[0])
        drug_matrix.append(l[1])
        line += 1
        # i += 1
        #
        # if i == Total_data_number:
        #     break
print("lines read ",line)
print(len(hsa_matrix)," " ,len(set(hsa_matrix)))
print(len(drug_matrix)," " ,len(set(drug_matrix)))


print("writing to file  ...")
# target = open("/home/farshid/NetBeansProjects/THesis/nr_network.txt", 'w')


counter = 0
total = 0
connection_list = []
# temp_list = []
for i in range(0,len(hsa_matrix)):
    temp_list = []
    for j in range(0,len(drug_matrix)):

        str1 = str(hsa_matrix[i])
        str2 = str(drug_matrix[j])

        # if temp_list.__contains__(str2):
        #     continue

        total += 1

        if i == j :
            counter += 1
            str3 = '1'
        else:
            str3 = '0'
        # temp_list.append(str1)
        temp_list.append(str2)
        # temp_list.append(str3)


        # connection_list.append(temp_list)
    # temp_list = []

        str4 = str1 + " "+ str2+" "+str3+"\n"
        # target.write(str4)
        connection_list.append( str4 )
# print(len(connection_list)," " ,len(set(connection_list)))

# list = sum((Counter(**{k:v}) for k , v in connection_list),Counter())

print(counter," and total ",total)

print(len(connection_list)," " ,len(set(connection_list)))

seen = set()
# new = []
new = [f for f in connection_list if f not in seen and not seen.add(x)]
connection_list = set(connection_list)
print(len(connection_list))

cls_one_cunter = 0

for l in connection_list:
    l = l.rstrip('\n').split(' ')
    # print(l)
    str1 = str(l[0])
    str2 = str(l[1])
    str3 = str(l[2])
    if l[2] == '1':
        cls_one_cunter +=1
    str4 = str1 + " " + str2 + " " + str3 + "\n"
    # target.write(str4)

# target.close()
print(" class one ", cls_one_cunter)