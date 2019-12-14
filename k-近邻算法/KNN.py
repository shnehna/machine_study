from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler


def knncls():
    """
    预测用户签到位置
    :return: None
    """
    # 读取数据
    data = pd.read_csv('./data/train.csv')
    # print(data.head(10))
    # 处理数据
    # 1、缩小数据 x y
    data = data.query('x>1.0 & x<1.25 & y>2.5 & y<2.75')
    # 2、处理时间
    time_value = pd.to_datetime(data['time'], unit='s')
    # print(time_value)
    time_value = pd.DatetimeIndex(time_value)
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday
    # pandas 0是行，1是列 sklearn与之相反
    data = data.drop(['time'], axis=1)
    # 把签到数量少于N个目标位置删除
    place_count = data.groupby('place_id').count()
    # 将place_id作为新的列 索引变成0-N
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]
    # 去除特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)
    x = data.drop(['row_id'], axis=1)
    # 数据分割 训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程（标准化）
    std = StandardScaler()
    # 对测试集和训练集的特征值标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    # 算法流程
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    print("预测目标签到位置：", y_predict)
    # 得出准确率
    print("预测准确率：", knn.score(x_test, y_test))


if __name__ == '__main__':
    knncls()
