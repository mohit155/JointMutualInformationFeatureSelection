import pandas
import math
import ast
import numpy
# from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn import datasets
# from sklearn.feature_selection import mutual_info_classif


def read(path):
    print('Reading...')
    if path.endswith('.csv'):
        data = pandas.read_csv(path)
    else:
        data = pandas.read_excel(path)
    data = data.dropna(axis=1, how='all')
    data = data._get_numeric_data()
    data = data.dropna()
    print("Data set read. Size =", data.shape)
    return data


def read_probability(path, cols):
    print('Reading...')
    file = open(path, 'r')
    prob = [{} for i in range(0, 2 * cols + 1)]
    col = 0
    for line in file.readlines():
        prob[col] = ast.literal_eval(line)
        col += 1
    file.close()
    return prob


def segment(data, k):
    y = data.values[:, k]
    x = data.drop(data.columns[k], axis=1)
    print(y.shape, x.shape)
    return x, y


def subset(data, score, threshold: any):
    x = []
    if threshold > 1:
        for i in range(0, threshold):
            x.append(data.values[:, score[i][0]])
    elif threshold > 0:
        s1 = 0
        s = 0
        for i in range(0, len(score)):
            s += score[i][1]
        for i in range(0, len(score)):
            s1 += score[i][1]
            x.append(data.values[:, score[i][0]])
            if s1 >= threshold*s:
                break
    x = numpy.array(x)
    x = x.transpose()
    x = pandas.DataFrame(x)
    x.columns = [data.columns[score[i][0]] for i in range(0, x.shape[1])]
    return x


def probability(data, y, filename):
    p = [{} for i in range(0, 2 * data.shape[1] + 1)]
    for i in range(0, data.shape[0]):
        print("\r", i, "/", data.shape[0], end='', flush=True)
        for j in range(0, data.shape[1]):
            if data.values[i][j] not in p[j]:
                p[j][data.values[i][j]] = 1 / len(data)
            else:
                p[j][data.values[i][j]] += 1 / len(data)
            if (data.values[i][j], y[i]) not in p[data.shape[1]+j+1]:
                p[data.shape[1] + j + 1][(data.values[i][j], y[i])] = 1 / len(data)
            else:
                p[data.shape[1] + j + 1][(data.values[i][j], y[i])] += 1 / len(data)
        if y[i] not in p[data.shape[1]]:
            p[data.shape[1]][y[i]] = 1 / len(data)
        else:
            p[data.shape[1]][y[i]] += 1 / len(data)

    file = open(filename, 'w')
    for i in range(0, 2 * data.shape[1] + 1):
        file.write(str(p[i]))
        file.write('\n')
    file.close()
    return p


def conditional_entropy(p, i, j, max_iter):
    result = 0
    count = 0
    flag = 0
    for val in p[i]:
        count += 1
        count1 = 0
        for val1 in p[j]:
            count1 += 1
            result += p[i][val]*p[j][val1]*math.log(p[i][val])
            if count1 == max_iter:
                break
        if count == max_iter:
            flag = 1
            break
    return -1*(result/max_iter) if flag == 1 else -1*(result/len(p[i]))


def joint_entropy(p, i, j, max_iter):
    result = 0
    count = 0
    flag = 0
    for val in p[i]:
        count += 1
        count1 = 0
        for val1 in p[j]:
            count1 += 1
            result += p[i][val]*p[j][val1]*math.log(p[i][val]*p[j][val1])
            if count1 == max_iter:
                break
        if count == max_iter:
            flag = 1
            break
    return -1*(result/max_iter) if flag == 1else -1*(result/len(p[i]))


def vif(data):
    print("Calculating Variance Inflation Factor.")
    vif = pandas.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif["Features"] = data.columns
    print(vif)
    return vif


def jmifs(x, y, filename, beta=1, reuse=False):
    # relevancy_score = [0]*x.shape[1]
    relevancy_score = [0 for i in range(0, x.shape[1])]
    for i in range(0, x.shape[1]):
        relevancy_score[i] = normalized_mutual_info_score(x.values[:, i], y)
    print(relevancy_score)

    # redundancy_score = [[0]*x.shape[1]]*x.shape[1]
    redundancy_score = [0 for i in range(0, x.shape[1])]

    for i in range(0, x.shape[1]):
        for j in range(0, x.shape[1]):
            redundancy_score[i] += normalized_mutual_info_score(x.values[:, i], x.values[:, j])
        redundancy_score[i] /= x.shape[1]
    print(redundancy_score)

    jmi_score = [0 for i in range(0, x.shape[1])]

    if reuse is not True:
        prob = probability(x, y, filename)
    else:
        prob = read_probability('./'+filename, x.shape[1])
    print("Processing...")
    for i in range(0, x.shape[1]):
        print('\r', i, '/', x.shape[1], end='', flush=True)
        for j in range(0, x.shape[1]):
            ixjy = conditional_entropy(prob, i, x.shape[1], 5000) - joint_entropy(prob, x.shape[1] + i, j, 5000)
            jmi_score[i] += ixjy + relevancy_score[j]
        jmi_score[i] /= x.shape[1]
    print()
    print(jmi_score)
    score = [0 for i in range(0, x.shape[1])]
    for i in range(0, x.shape[1]):
        score[i] = relevancy_score[i] + -1*beta*redundancy_score[i] + jmi_score[i]
    print(score)
    return score


# Data = read('./DataSet/kc_house_data.csv')
# Data, Y = segment(Data, 1)
Data, Y = datasets.load_breast_cancer(return_X_y=True)
# Data, Y = datasets.load_iris(return_X_y=True)
Data = pandas.DataFrame(Data)
jmi = jmifs(Data, Y, 'Prob_Breast_Cancer.txt', reuse=True)
jmi_dict = [[i, jmi[i]] for i in range(0, len(jmi))]
jmi_dict = sorted(jmi_dict, key=lambda kv: kv[1], reverse=True)
print(jmi_dict)
vif_pre = vif(Data)
vif_post = vif(subset(Data, jmi_dict, math.ceil(Data.shape[1]/2)))
plt.bar([i-0.1 for i in range(0, vif_pre.shape[0])], vif_pre["VIF Factor"], width=0.2, color='r', align='center')
print(vif_pre.columns.tolist())
plt.bar([vif_pre.values[:, 1].tolist().index(vif_post.values[i][1]) + 0.1 for i in range(0, vif_post.shape[0])],
        vif_post["VIF Factor"], width=0.2, color='b', align='center')
plt.xticks([i for i in range(0, vif_pre.shape[0])])
plt.xlabel("Features")
plt.ylabel("VIF")
plt.title("Variance Inflation Factor vs Feature")
plt.legend(['Initial VIF Score', 'New VIF Score'], loc=1)
plt.savefig('../VIF Breast_Cancer_2')
plt.show()
# vif_house_post1 = vif(subset(housing_data, jmi_dict, 0.9))
