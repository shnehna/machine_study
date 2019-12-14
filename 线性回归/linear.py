from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def mylinear():
    # 获取数据
    lb = load_boston()
    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    # print(y_train, y_test)
    # 标准化处理(特征值和目标值都要标准化,而且是两个实例化对象)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.fit_transform(x_test)
    std_y = StandardScaler()
    # 19版本标准化要求参数为 2维数组 numpy中的reshape()方法
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.fit_transform(y_test.reshape(-1, 1))
    # 预测
    # 正规方程
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(lr.coef_)
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    print("每个房子预测价格", y_lr_predict)
    print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))
    # 梯度下降
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print(sgd.coef_)
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    print("每个房子预测价格", y_sgd_predict)
    print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))


if __name__ == '__main__':
    mylinear()
