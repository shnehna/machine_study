from sklearn.preprocessing import MinMaxScaler


def mn():
    """
    归一化处理
    :return: None
    """
    mn = MinMaxScaler(feature_range=(0, 1))
    data = mn.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)
    return None


if __name__ == '__main__':
    mn()
