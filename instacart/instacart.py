import pandas as pd
from sklearn.decomposition import PCA

# 读取四张表数据
prior = pd.read_csv("./data/order_products__prior.csv")
products = pd.read_csv("./data/products.csv")
orders = pd.read_csv("./data/orders.csv")
aisles = pd.read_csv("./data/aisles.csv")
# 合并到一张表中
_mg = pd.merge(prior, products, on=['product_id', 'product_id'])
_mg = pd.merge(_mg, orders, on=['order_id', 'order_id'])
mt = pd.merge(_mg, aisles, on=['aisle_id', 'aisle_id'])
mt.head(10)
cross = pd.crosstab(mt['user_id'], mt['aisle'])
cross.head(10)
pca = PCA(n_components=0.9)
data = pca.fit_transform(cross)
data.shape
