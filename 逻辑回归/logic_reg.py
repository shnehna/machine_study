import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def logistic():
    column = [
        'Sample code number',
        'Clump Thickness',
        'Uniformity of Cell Size',
        'Uniformity of Cell Shape',
        'Marginal Adhesion',
        'Single Epithelial Cell Size',
        'Bare Nuclei',
        'Bland Chromatin',
        'Normal Nucleoli',
        'Mitoses',
        'Class'
    ]
    data = pd.read_csv("./breast-cancer-wisconsin.data", names=column)
    # print(data)
    # 缺失值处理
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna()

    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

    # 标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)
    print(lg.coef_)
    y_predict = lg.predict(x_test)
    print("准确率：", lg.score(x_test, y_test))
    # 召回率
    print("召回率：", classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性']))


if __name__ == '__main__':
    logistic()
