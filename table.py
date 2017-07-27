with open('/home/farshid/Desktop/enzyme_dataset_1477.txt', 'r') as file_read:
    for x in file_read:
        if len(x) <= 10:
            break
        l = x.rstrip('\n').split(',')

        print(l[1282:])
        # break

with open('/home/farshid/Desktop/enzyme_dataset_1294.txt', 'r') as file_read:
    for x in file_read:
        if len(x) <= 10:
            break
        l = x.rstrip('\n').split(',')

        print(l[1282:])
        break