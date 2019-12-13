from sklearn.decomposition import PCA


def pca_vec():
    pc = PCA(n_components=0.9)
    print(pc.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]))


# n_components 表示数据损失后比例  建议0.9-0.95

if __name__ == '__main__':
    pca_vec()
