# """
# ===========================
# Neighbourhood Cleaning Rule
# ===========================
#
# An illustration of the neighbourhood cleaning rule method.
#
# """
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.datasets import make_classification
# from sklearn.decomposition import PCA
# import standardalize
#
from imblearn.under_sampling import NeighbourhoodCleaningRule


#
# # print(__doc__)
#
# total_matrix = [[]]
# feature_list_of_all_instances = []
# class_list_of_all_instances = []
# total_matrix = []
# Total_data_number = int(291920/2)
# # Total_data_number = 5000
# # clf = MLPClassifier(hidden_layer_sizes=5000,activation='logistic',learning_rate='adaptive')
# data = []  # this list is to generate index value for k fold validation
#
#
#
# # file_read = open('/home/farshid/Desktop/dataset.txt', 'r')
# count_for_number_of_instances = 0
# i = 0
#
# # for i in range(0, 1287):
# #     file_read.readline()
# print("Starting To read From Text ...  ")
# with open('dataset_test.txt', 'r') as file_read:
#     for x in file_read:
#
#         # x = file_read.readline()
#         if len(x) <= 10:
#             break
#         l = x.split(",")
#         # l = list(map(float, l))
#         total_matrix.append(l)
#         # if l[1294] == 1:
#         #     i += 1
#
#         if i == Total_data_number :
#             break
#
#
# print("Starting To Standardize Total Matrix(",i,") ...  ")
# # total_matrix = standardalize.std(total_matrix, 882, 412)
#
# for l in total_matrix:
#     feature_list_of_all_instances.append(l[0:1294])
#     class_list_of_all_instances.append(int(l[1294]))


# sns.set()
#
# Define some color for the plotting
# almost_black = '#262626'
# palette = sns.color_palette()


# Generate the dataset
# X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
#                            n_informative=10, n_redundant=2, flip_y=0,
#                            n_repeated=7,
#                            n_features=20, n_clusters_per_class=2,
#                            n_samples=5000, random_state=10)

# Instanciate a PCA object for the sake of easy visualisation
# pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
# X_vis = pca.fit_transform(X)

def neighbourhood_cleaning_rule(feature_list_of_all_instances, class_list_of_all_instances,neighbours):
    # Apply neighbourhood cleaning rule
    c1 = 0
    c2 = 0
    count = 0
    for i in class_list_of_all_instances:
        if i == 1:
            c1 += 1
        if i == 0:
            c2 += 1
        if i != 1 and i != 0:
            count += 1

    print("     Data of class 1 ", c1, " ,Data of cls 0 ", c2, ",Other class ", count)

    # for i in range(5,200,5):

    ncl = NeighbourhoodCleaningRule(n_neighbors=neighbours, n_jobs=4)
    X_resampled, y_resampled = ncl.fit_sample(feature_list_of_all_instances, class_list_of_all_instances)
    # X_res_vis = pca.transform(X_resampled)
    # 13
    print("     Cleaned ", len(feature_list_of_all_instances) - len(X_resampled), " points", end='')

    c1 = 0
    c2 = 0
    for ii in y_resampled:
        if ii == 1:
            c1 += 1
        if ii == 0:
            c2 += 1

    print(" and data of class 1 ", c1, "data of cls 0 ", c2, "for ", neighbours, "neighbours ")


    return X_resampled, y_resampled  # feature_list_of_all_instances,class_list_of_all_instances=neighbourhood_cleaning_rule(feature_list_of_all_instances,class_list_of_all_instances)
# print("length of total data" , len(feature_list_of_all_instances))
#
# # Two subplots, unpack the axes array immediately
# f, (ax1, ax2) = plt.subplots(1, 2)
#
# ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
#             edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
# ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5,
#             edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
# ax1.set_title('Original set')
#
# ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
#             label="Class #0", alpha=.5, edgecolor=almost_black,
#             facecolor=palette[0], linewidth=0.15)
# ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
#             label="Class #1", alpha=.5, edgecolor=almost_black,
#             facecolor=palette[2], linewidth=0.15)
# ax2.set_title('Neighbourhood cleaning rule')
#
# # plt.show()
