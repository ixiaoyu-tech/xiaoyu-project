import pandas
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import model_selection  # 模型比较和选择包
from sklearn.naive_bayes import GaussianNB


class Bayes_Test():

    # 读取样本 数据集
    def load_dataset(self):
        url = 'wine.csv'
        names = ['class', 'Al', 'Ma', 'Ash', 'Aoa', 'Mag', 'Top', 'Fl', 'No',
                 'Pr', 'Co', 'Hue', 'OD', 'Pro']
        dataset = pandas.read_csv(url, names=names)
        return dataset

    # 提取样本特征集和类别集 划分训练/测试集
    def split_out_dataset(self, dataset):
        array = dataset.values  # 将数据库转换成数组形式
        X = array[:, 1:14].astype(float)  # 取特征数值列
        Y = array[:, 0]  # 取类别列
        validation_size = 0.20  # 验证集规模
        seed = 7
        # 分割数据集 测试/验证
        X_train, X_validation, Y_train, Y_validation = \
            model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
        return X_train, X_validation, Y_train, Y_validation

    """第一步：划分样本集"""

    # 提取样本 特征
    def split_out_attributes(self, X, Y):
        # 提取 每个类别的不同特征
        # c1 第一类的特征数组
        c1_1 = c1_2 = c1_3 = c1_4 = c1_5 = c1_6 = c1_7 = c1_8 = c1_9 = c1_10 = c1_11 = c1_12 = c1_13 = []
        c2_1 = c2_2 = c2_3 = c2_4 = c2_5 = c2_6 = c2_7 = c2_8 = c2_9 = c2_10 = c2_11 = c2_12 = c2_13 = []
        c3_1 = c3_2 = c3_3 = c3_4 = c3_5 = c3_6 = c3_7 = c3_8 = c3_9 = c3_10 = c3_11 = c3_12 = c3_13 = []
        for i in range(len(Y)):
            if (Y[i] == 1):
                c1_1.append(X[i, 0])
                c1_2.append(X[i, 1])
                c1_3.append(X[i, 2])
                c1_4.append(X[i, 3])
                c1_5.append(X[i, 4])
                c1_6.append(X[i, 5])
                c1_7.append(X[i, 6])
                c1_8.append(X[i, 7])
                c1_9.append(X[i, 8])
                c1_10.append(X[i, 9])
                c1_11.append(X[i, 10])
                c1_12.append(X[i, 11])
                c1_13.append(X[i, 12])
            elif (Y[i] == 2):
                # c2 第二类的特征数组
                c2_1.append(X[i, 0])
                c2_2.append(X[i, 1])
                c2_3.append(X[i, 2])
                c2_4.append(X[i, 3])
                c2_5.append(X[i, 4])
                c2_6.append(X[i, 5])
                c2_7.append(X[i, 6])
                c2_8.append(X[i, 7])
                c2_9.append(X[i, 8])
                c2_10.append(X[i, 9])
                c2_11.append(X[i, 10])
                c2_12.append(X[i, 11])
                c2_13.append(X[i, 12])
            elif (Y[i] == 3):
                # c3 第三类的特征数组
                c3_1.append(X[i, 0])
                c3_2.append(X[i, 1])
                c3_3.append(X[i, 2])
                c3_4.append(X[i, 3])
                c3_5.append(X[i, 4])
                c3_6.append(X[i, 5])
                c3_7.append(X[i, 6])
                c3_8.append(X[i, 7])
                c3_9.append(X[i, 8])
                c3_10.append(X[i, 9])
                c3_11.append(X[i, 10])
                c3_12.append(X[i, 11])
                c3_13.append(X[i, 12])
            else:
                pass

        return [c1_1, c1_2, c1_3, c1_4, c1_5, c1_6, c1_7, c1_8, c1_9, c1_10, c1_11, c1_12, c1_13,
                c2_1, c2_2, c2_3, c2_4, c2_5, c2_6, c2_7, c2_8, c2_9, c2_10, c2_11, c2_12, c2_13,
                c3_1, c3_2, c3_3, c3_4, c3_5, c3_6, c3_7, c3_8, c3_9, c3_10, c3_11, c3_12, c3_13]

    """因为符合多变量正态分布，所以需要(μ，∑)两个样本参数"""
    """第二步：计算样本期望μ和样本方差s"""

    # 计算样本期望
    def cal_mean(self, attributes):
        c1_1, c1_2, c1_3, c1_4, c1_5, c1_6, c1_7, c1_8, c1_9, c1_10, c1_11, c1_12, c1_13 \
            , c2_1, c2_2, c2_3, c2_4, c2_5, c2_6, c2_7, c2_8, c2_9, c2_10, c2_11, c2_12, c2_13 \
            , c3_1, c3_2, c3_3, c3_4, c3_5, c3_6, c3_7, c3_8, c3_9, c3_10, c3_11, c3_12, c3_13 = attributes

        # 第一类的期望值μ
        e_c1_1 = np.mean(c1_1)
        e_c1_2 = np.mean(c1_2)
        e_c1_3 = np.mean(c1_3)
        e_c1_4 = np.mean(c1_4)
        e_c1_5 = np.mean(c1_5)
        e_c1_6 = np.mean(c1_6)
        e_c1_7 = np.mean(c1_7)
        e_c1_8 = np.mean(c1_8)
        e_c1_9 = np.mean(c1_9)
        e_c1_10 = np.mean(c1_10)
        e_c1_11 = np.mean(c1_11)
        e_c1_12 = np.mean(c1_12)
        e_c1_13 = np.mean(c1_13)
        # 第二类的期望值μ
        e_c2_1 = np.mean(c2_1)
        e_c2_2 = np.mean(c2_2)
        e_c2_3 = np.mean(c2_3)
        e_c2_4 = np.mean(c2_4)
        e_c2_5 = np.mean(c2_5)
        e_c2_6 = np.mean(c2_6)
        e_c2_7 = np.mean(c2_7)
        e_c2_8 = np.mean(c2_8)
        e_c2_9 = np.mean(c2_9)
        e_c2_10 = np.mean(c2_10)
        e_c2_11 = np.mean(c2_11)
        e_c2_12 = np.mean(c2_12)
        e_c2_13 = np.mean(c2_13)
        # 第三类的期望值μ
        e_c3_1 = np.mean(c3_1)
        e_c3_2 = np.mean(c3_2)
        e_c3_3 = np.mean(c3_3)
        e_c3_4 = np.mean(c3_4)
        e_c3_5 = np.mean(c3_5)
        e_c3_6 = np.mean(c3_6)
        e_c3_7 = np.mean(c3_7)
        e_c3_8 = np.mean(c3_8)
        e_c3_9 = np.mean(c3_9)
        e_c3_10 = np.mean(c3_10)
        e_c3_11 = np.mean(c3_11)
        e_c3_12 = np.mean(c3_12)
        e_c3_13 = np.mean(c3_13)

        return [e_c1_1, e_c1_2, e_c1_3, e_c1_4, e_c1_5, e_c1_6, e_c1_7, e_c1_8, e_c1_9, e_c1_10, e_c1_11, e_c1_12,
                e_c1_13,
                e_c2_1, e_c2_2, e_c2_3, e_c2_4, e_c2_5, e_c2_6, e_c2_7, e_c2_8, e_c2_9, e_c2_10, e_c2_11, e_c2_12,
                e_c2_13,
                e_c3_1, e_c3_2, e_c3_3, e_c3_4, e_c3_5, e_c3_6, e_c3_7, e_c3_8, e_c3_9, e_c3_10, e_c3_11, e_c3_12,
                e_c3_13]

    # 计算样本方差
    def cal_var(self, attributes):
        c1_1, c1_2, c1_3, c1_4, c1_5, c1_6, c1_7, c1_8, c1_9, c1_10, c1_11, c1_12, c1_13 \
            , c2_1, c2_2, c2_3, c2_4, c2_5, c2_6, c2_7, c2_8, c2_9, c2_10, c2_11, c2_12, c2_13 \
            , c3_1, c3_2, c3_3, c3_4, c3_5, c3_6, c3_7, c3_8, c3_9, c3_10, c3_11, c3_12, c3_13 = attributes

        # 第一类的方差var
        var_c1_1 = np.var(c1_1)
        var_c1_2 = np.var(c1_2)
        var_c1_3 = np.var(c1_3)
        var_c1_4 = np.var(c1_4)
        var_c1_5 = np.var(c1_5)
        var_c1_6 = np.var(c1_6)
        var_c1_7 = np.var(c1_7)
        var_c1_8 = np.var(c1_8)
        var_c1_9 = np.var(c1_9)
        var_c1_10 = np.var(c1_10)
        var_c1_11 = np.var(c1_11)
        var_c1_12 = np.var(c1_12)
        var_c1_13 = np.var(c1_13)
        # 第二类的方差s
        var_c2_1 = np.var(c2_1)
        var_c2_2 = np.var(c2_2)
        var_c2_3 = np.var(c2_3)
        var_c2_4 = np.var(c2_4)
        var_c2_5 = np.var(c2_5)
        var_c2_6 = np.var(c2_6)
        var_c2_7 = np.var(c2_7)
        var_c2_8 = np.var(c2_8)
        var_c2_9 = np.var(c2_9)
        var_c2_10 = np.var(c2_10)
        var_c2_11 = np.var(c2_11)
        var_c2_12 = np.var(c2_12)
        var_c2_13 = np.var(c2_13)
        # 第三类的方差s
        var_c3_1 = np.var(c3_1)
        var_c3_2 = np.var(c3_2)
        var_c3_3 = np.var(c3_3)
        var_c3_4 = np.var(c3_4)
        var_c3_5 = np.var(c3_5)
        var_c3_6 = np.var(c3_6)
        var_c3_7 = np.var(c3_7)
        var_c3_8 = np.var(c3_8)
        var_c3_9 = np.var(c3_9)
        var_c3_10 = np.var(c3_10)
        var_c3_11 = np.var(c3_11)
        var_c3_12 = np.var(c3_12)
        var_c3_13 = np.var(c3_13)

        return [var_c1_1, var_c1_2, var_c1_3, var_c1_4, var_c1_5, var_c1_6, var_c1_7, var_c1_8, var_c1_9, var_c1_10,
                var_c1_11, var_c1_12, var_c1_13,
                var_c2_1, var_c2_2, var_c2_3, var_c2_4, var_c2_5, var_c2_6, var_c2_7, var_c2_8, var_c2_9, var_c2_10,
                var_c2_11, var_c2_12, var_c2_13,
                var_c3_1, var_c3_2, var_c3_3, var_c3_4, var_c3_5, var_c3_6, var_c3_7, var_c3_8, var_c3_9, var_c3_10,
                var_c3_11, var_c3_12, var_c3_13]

    # 计算先验概率P(Y=ck)
    def cal_prior_probability(self, Y):
        a = b = c = 0

        for i in Y:
            if (i == 1):
                a += 1
            elif (i == 2):
                b += 1
            elif (i == 3):
                c += 1
            else:
                pass

        pa = a / len(Y)
        pb = b / len(Y)
        pc = c / len(Y)
        return pa, pb, pc

    # 计算后验概率P(Y=ck|X)=P(X|Y=ck)*P(Y=ck)/∑
    def cal_posteriori_probability(self, X, Y, p, means, vars):
        pa, pb, pc = p

        e_c1_1, e_c1_2, e_c1_3, e_c1_4, e_c1_5, e_c1_6, e_c1_7, e_c1_8, e_c1_9, e_c1_10, e_c1_11, e_c1_12, e_c1_13 \
            , e_c2_1, e_c2_2, e_c2_3, e_c2_4, e_c2_5, e_c2_6, e_c2_7, e_c2_8, e_c2_9, e_c2_10, e_c2_11, e_c2_12, e_c2_13 \
            , e_c3_1, e_c3_2, e_c3_3, e_c3_4, e_c3_5, e_c3_6, e_c3_7, e_c3_8, e_c3_9, e_c3_10, e_c3_11, e_c3_12, e_c3_13 = means

        var_c1_1, var_c1_2, var_c1_3, var_c1_4, var_c1_5, var_c1_6, var_c1_7, var_c1_8, var_c1_9, var_c1_10, var_c1_11, var_c1_12, var_c1_13 \
            , var_c2_1, var_c2_2, var_c2_3, var_c2_4, var_c2_5, var_c2_6, var_c2_7, var_c2_8, var_c2_9, var_c2_10, var_c2_11, var_c2_12, var_c2_13 \
            , var_c3_1, var_c3_2, var_c3_3, var_c3_4, var_c3_5, var_c3_6, var_c3_7, var_c3_8, var_c3_9, var_c3_10, var_c3_11, var_c3_12, var_c3_13 = vars

        print('p:', p)
        print('means:', means)
        print('vars:', vars)

        # 分解十三维输入向量X=[X1，X2，X3...X13]为13个一维正态分布函数
        X1 = X[:, 0]
        X2 = X[:, 1]
        X3 = X[:, 2]
        X4 = X[:, 3]
        X5 = X[:, 4]
        X6 = X[:, 5]
        X7 = X[:, 6]
        X8 = X[:, 7]
        X9 = X[:, 8]
        X10 = X[:, 9]
        X11 = X[:, 10]
        X12 = X[:, 11]
        X13 = X[:, 12]

        # 分类正确数/分类错误数=>计算正确率
        true_test = 0
        false_test = 0

        # 遍历训练整个输入空间，计算后验概率并判决
        for i in range(len(X1)):
            # 计算后验概率=P(X|Y=C1)P(Y=C1)
            P_1 = stats.norm.pdf(
                X1[i], e_c1_1, var_c1_1) * stats.norm.pdf(
                X2[i], e_c1_2, var_c1_2) * stats.norm.pdf(
                X3[i], e_c1_3, var_c1_3) * stats.norm.pdf(
                X4[i], e_c1_4, var_c1_4) * stats.norm.pdf(
                X5[i], e_c1_5, var_c1_5) * stats.norm.pdf(
                X6[i], e_c1_6, var_c1_6) * stats.norm.pdf(
                X7[i], e_c1_7, var_c1_7) * stats.norm.pdf(
                X8[i], e_c1_8, var_c1_8) * stats.norm.pdf(
                X9[i], e_c1_9, var_c1_9) * stats.norm.pdf(
                X10[i], e_c1_10, var_c1_10) * stats.norm.pdf(
                X11[i], e_c1_11, var_c1_11) * stats.norm.pdf(
                X12[i], e_c1_12, var_c1_12) * stats.norm.pdf(
                X13[i], e_c1_13, var_c1_13) * pa
            # 计算后验概率=P(X|Y=C2)P(Y=C2)
            P_2 = stats.norm.pdf(
                X1[i], e_c2_1, var_c2_1) * stats.norm.pdf(
                X2[i], e_c2_2, var_c2_2) * stats.norm.pdf(
                X3[i], e_c2_3, var_c2_3) * stats.norm.pdf(
                X4[i], e_c2_4, var_c2_4) * stats.norm.pdf(
                X5[i], e_c2_5, var_c2_5) * stats.norm.pdf(
                X6[i], e_c2_6, var_c2_6) * stats.norm.pdf(
                X7[i], e_c2_7, var_c2_7) * stats.norm.pdf(
                X8[i], e_c2_8, var_c2_8) * stats.norm.pdf(
                X9[i], e_c2_9, var_c2_9) * stats.norm.pdf(
                X10[i], e_c2_10, var_c2_10) * stats.norm.pdf(
                X11[i], e_c2_11, var_c2_11) * stats.norm.pdf(
                X12[i], e_c2_12, var_c2_12) * stats.norm.pdf(
                X13[i], e_c2_13, var_c2_13) * pb
            # 计算后验概率=P(X|Y=C3)P(Y=C3)
            P_3 = stats.norm.pdf(
                X1[i], e_c3_1, var_c3_1) * stats.norm.pdf(
                X2[i], e_c3_2, var_c3_2) * stats.norm.pdf(
                X3[i], e_c3_3, var_c3_3) * stats.norm.pdf(
                X4[i], e_c3_4, var_c3_4) * stats.norm.pdf(
                X5[i], e_c3_5, var_c3_5) * stats.norm.pdf(
                X6[i], e_c3_6, var_c3_6) * stats.norm.pdf(
                X7[i], e_c3_7, var_c3_7) * stats.norm.pdf(
                X8[i], e_c3_8, var_c3_8) * stats.norm.pdf(
                X9[i], e_c3_9, var_c3_9) * stats.norm.pdf(
                X10[i], e_c3_10, var_c3_10) * stats.norm.pdf(
                X11[i], e_c3_11, var_c3_11) * stats.norm.pdf(
                X12[i], e_c3_12, var_c3_12) * stats.norm.pdf(
                X13[i], e_c3_13, var_c3_13) * pc

            # 计算判别函数，选取概率最大的类
            max_P = max(P_1, P_2, P_3)
            # 输出分类结果，并检测正确率
            if (max_P == P_1):
                if (Y[i] == 1):
                    print('分为第一类，正确')
                    true_test += 1
                else:
                    print('分为第一类，错误')
                    false_test += 1
            elif (max_P == P_2):
                if (Y[i] == 2):
                    print('分为第二类，正确')
                    true_test += 1
                else:
                    print('分为第二类，错误')
                    false_test += 1
            elif (max_P == P_3):
                if (Y[i] == 3):
                    print('分为第三类，正确')
                    true_test += 1
                else:
                    print('分为第三类，错误')
                    false_test += 1
            else:
                print('未分类')
                false_test += 1
        # 打印分类正确率
        print('训练正确率为:', (true_test / (true_test + false_test)))

        # 模板方法对照

    def cal_dataset(self, X_train, Y_train):
        # Test options and evaluation metric
        seed = 7
        scoring = 'accuracy'
        # Check Algorithms
        model = GaussianNB()
        name = 'bayes classifier'
        # 建立K折交叉验证 10倍
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        # cross_val_score() 对数据集进行指定次数的交叉验证并为每次验证效果评测
        cv_results = \
            model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results = cv_results
        msg = "%s: %f (%f)" % (name + '精度', cv_results.mean(), cv_results.std())
        print(msg)

        # Show Algorithms
        dataresult = pandas.DataFrame(results)
        dataresult.plot(title='Bayes accuracy analysis', kind='density', subplots=True, layout=(1, 1), sharex=False,
                        sharey=False)
        dataresult.hist()
        plt.show()

bayes = Bayes_Test()

dataset = bayes.load_dataset()
# 划分训练集 测试集
X_train, X_validation, Y_train, Y_validation = bayes.split_out_dataset(dataset)
print('得到的X_train', X_train)
print('得到的Y_train', Y_train)

# 分割属性--训练集
attributes = bayes.split_out_attributes(X_train, Y_train)
print('得到的训练集', attributes)

# 计算期望--训练集
means = bayes.cal_mean(attributes)
print('得到的means', means)

# 计算方差--训练集
vars = bayes.cal_var(attributes)
print('得到的vars', vars)

# 计算先验概率--训练集
prior_p = bayes.cal_prior_probability(Y_train)
# 验证分类准确性--测试集
bayes.cal_posteriori_probability(X_train, Y_train, prior_p, means, vars)

# 模板方法--性能对比
bayes.cal_dataset(X_validation,Y_validation)