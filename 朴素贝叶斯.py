# tf: trem frequency 词频
# idf: inverse document frequency 逆文档频率  log(总文档数量/该词出现的文档数)
# tf * idf 重要性程度    浮点型数值
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer


def cut_word():
    cut1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝大部分是死在明天晚上，所以每个人不要放弃今天。")
    cut2 = jieba.cut("我们看到的从很远星系来的光是几百万年之前发出的，这样当我们看到宇宙时，我们是在看他的过去。")
    cut3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解他，了解是我的真正含义的秘密取决于如何将其与我们所了解的事物相联系。")
    con1 = list(cut1)
    con2 = list(cut2)
    con3 = list(cut3)
    c1 = " ".join(con1)
    c2 = " ".join(con2)
    c3 = " ".join(con3)
    return c1, c2, c3


def tf_idf_vec():
    """中文特征值化"""
    c1, c2, c3 = cut_word()
    tf = TfidfVectorizer()
    data = tf.fit_transform([c1, c2, c3])
    print(tf.get_feature_names())
    print(data.toarray())


if __name__ == '__main__':
    tf_idf_vec()
