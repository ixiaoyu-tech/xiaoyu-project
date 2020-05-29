from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz



def wine_tree():
    """
    用决策树对红酒的数据集进行分类。
    """
    #获取数据集
    wine = load_wine()

    # 数据集打乱划分为训练集和测试机

    x_train, x_test, y_train, y_test = train_test_split(wine['data'],
                                                        wine['target'],
                                                        test_size=0.25,
                                                        random_state=14)
    # 决策树的模型：
    estimator = DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train,y_train)

    #模型评估：
    # <1>直接对比真实值与预测值
    y_pred = estimator.predict(x_test)  # 1 X 20 向量
    print("预测值为:\n {}", y_pred)
    print("真实值和预测值", y_test == y_pred)
    # <2>计算准确率：
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    #可视化决策树：
    export_graphviz(estimator,out_file= 'wine_tree.dot',feature_names=wine.feature_names)
    return None

if __name__ == '__main__':
    wine_tree()