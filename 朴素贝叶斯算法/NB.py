from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def naviebayes():
    news = fetch_20newsgroups(data_home='./', subset='all')
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
    # 对数据集进行特征抽取
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    # print(tf.get_feature_names())
    x_test = tf.transform(x_test)
    # 进行朴素贝叶斯算法
    nb = MultinomialNB(alpha=1.0)
    # print(x_train)
    nb.fit(x_train, y_train)
    y_predict = nb.predict(x_test)
    # print("预测文章类别为：", y_predict)
    # print("准确率：", nb.score(x_test, y_test))

    print("每个类别的精确率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))


if __name__ == '__main__':
    naviebayes()
