# 导入数据
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
# noinspection PyUnresolvedReferences
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# noinspection PyUnresolvedReferences
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def knn_wine_gscv():
        #添加网格搜索和交叉验证
        wine_dataset = load_wine()

        # 数据集打乱划分为训练集和测试机

        X_train, X_test, y_train, y_test = train_test_split(wine_dataset['data'],
                                                            wine_dataset['target'],
                                                            test_size=0.25,
                                                            random_state=14)
        # 标准化处理：
        transfer = StandardScaler()
        X_train = transfer.fit_transform(X_train)
        X_test = transfer.transform(X_test)


        estimator = KNeighborsClassifier()
        #加入网格搜索和交叉验证：
        estimator = GridSearchCV(estimator = estimator,param_grid={'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]},cv= 10)
        estimator.fit(X_train, y_train)

        # <1>直接对比真实值与预测值
        y_pred = estimator.predict(X_test)  # 1 X 20 向量
        print("预测值为:\n {}", y_pred)
        print("真实值和预测值", y_test == y_pred)
        # <2>计算准确率：
        score = estimator.score(X_test, y_test)
        print("准确率为：\n", score)
        # <3>查看最佳结果：
        print("最佳参数：\n",estimator.best_params_)
        print("最佳结果：\n", estimator.best_score_)
        print("最佳估计器：\n", estimator.best_estimator_)
        print("交叉验证结果：\n", estimator.cv_results_)
        return None

if __name__ == "__main__":
        knn_wine_gscv()
