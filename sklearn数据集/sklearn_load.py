from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
# print("特征值")
print(iris.data)
# print("目标值")
print(iris.target)
print(iris.DESCR)

# 返回值 训练集 和 测试集  x_train,y_train  x_test,y_test
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)
print("训练集的特征值和目标值：", x_train, y_train)
print("测试集的特征值和目标值：", x_test, y_test)

