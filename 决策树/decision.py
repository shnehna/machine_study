import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


def decision():
    """
    对泰坦尼克号乘客进行预测生死
    :return: None
    """
    # 获取数据
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 处理数据 找出特征值目标值
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']
    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)
    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程 特征->类别-> one_hot编码
    vec = DictVectorizer(sparse=False)
    x_train = vec.fit_transform(x_train.to_dict(orient="records", into=dict))
    x_test = vec.transform(x_test.to_dict(orient="records"))
    dec = DecisionTreeClassifier(max_depth=5)
    dec.fit(x_train, y_train)
    print(dec.score(x_test, y_test))
    # 到处决策树
    export_graphviz(dec, out_file="./tree.dot",
                    feature_names=['年龄', 'pclass=1st', 'pclass=2st', 'pclass=3st', '女性', '男性'])
    return None


if __name__ == '__main__':
    decision()
