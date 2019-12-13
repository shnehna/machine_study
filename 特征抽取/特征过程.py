from sklearn.feature_extraction import DictVectorizer


def dictvec():
    """
    字典数据抽取
    :return: None
    """
    # 实例化
    dicts = DictVectorizer()
    # 调用 fit_transform
    city_list = [
        {'city': '北京', 'temperature': 30},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 90}
    ]
    transform = dicts.fit_transform(city_list)
    print(dicts.get_feature_names())
    print(dicts.inverse_transform(transform))
    print(transform)


if __name__ == '__main__':
    dictvec()
