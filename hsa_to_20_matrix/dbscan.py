from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

iris_data , iris_class = load_iris(True)

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(algorithm='auto',eps=.9,min_samples=3)

X = StandardScaler().fit_transform(iris_data)
dbscan.fit(X)
# print(iris_data )
print( dbscan.labels_ )
error = 0
for i in dbscan.labels_:
    if i == -1 :
        error += 1
print("Total points " , 150)
print("Total outliers " ,error)
print("Clustered  Area ",(1-error/150 )* 100 ,"%")
# index_lits = []
# point = 0
# for index in range(0,150):
#     if iris_class[index] == 0 and point == 0 :
#         point += 1
#         index_lits.append(index)
#     if iris_class[index] == 1 and point == 1:
#         index_lits.append(index)
#         point += 1
#     if iris_class[index] == 2 and point == 2:
#         index_lits.append(index)
#         point += 1

# print(index_lits)
print(len(dbscan.labels_))
print(dbscan)
X = iris_data[149]

print("number of cluster : " , len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0))
# for index in range(0,150):
#     i = np.asarray( iris_data[index] )
#     print(dbscan.fit_predict([i]) ,end ="")
#     if index % 25 == 0 :
#         print()


# Y = iris_data[128].reshape(1,4)
#
iris_data = iris_data.tolist()

iris_data.append(iris_data[0])
iris_data = np.asarray(iris_data)

print(dbscan.fit_predict(iris_data) )
print("Total  " , len(dbscan.fit_predict(iris_data) ) )

# print(dbscan.fit_predict(Y) )

#