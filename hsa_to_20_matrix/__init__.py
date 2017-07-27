import re
import numpy as np
import mysql.connector



class do_stuff:
    hsa = 'hsa:10269.seq'
    path = '/home/farshid/PycharmProjects/Experiments_for_SS/hsa_to_20_matrix/' + hsa + '.hsa_to_20_matrix'
    file = open(path, 'r')
    file.readline()
    file.readline()
    final_list = []
    num_lines = sum(1 for line in open(path, 'r'))
    num_lines = num_lines - 6 - 3  # 6 from the bottom 3 from the top
    array = np.zeros(shape=(num_lines, 20))
    hsa_list = []
    matrix = [[0 for x in range(20)] for y in range(20)]

    def __init__(self):
        cnx = mysql.connector.connect(user='root', password='123',
                                      host='127.0.0.1',
                                      database='thesisDataset')
        cursor = cnx.cursor()

        query = ("SELECT hsa FROM hsaSeq")
        x = cursor.execute(query)

        for i in cursor:
            self.hsa_list.append(i)

        cnx.close()

    def do_shit(self):

        str = self.file.read(89)
        str = str[11:]
        a = []

        str = re.findall("-?\d+", str)

        self.file.readline()

        return str

    def create_final_list(self):

        self.do_shit()
        i = 0

        while i < do_stuff.num_lines:
            do_stuff.final_list.append(self.do_shit())
            i = i + 1

        do_stuff.array = do_stuff.final_list

    def get_matrix(self):
        for m in range(0, 20):
            for n in range(0, 20):
                for i in range(0, self.num_lines - 1):
                    self.matrix[m][n] += int(self.array[i][m]) * int(self.array[i + 1][n])

    def write(self, matrix):

        file = open(self.hsa + '.csv', 'w')
        for m in range(0, len(matrix)):
            for n in range(0, len(matrix[m])):
                file.write(str(matrix[m][n]) + ", ")
            file.write("\n")
        file.close()


do = do_stuff()

do.create_final_list()
do.get_matrix()
do.write(do.matrix)

