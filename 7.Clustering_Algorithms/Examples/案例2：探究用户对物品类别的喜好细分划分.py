# 0.导包
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1.获取数据
order_product = pd.read_csv("../../Data/instacart-market-basket-analysis/order_products__prior.csv")
products = pd.read_csv("../../Data/instacart-market-basket-analysis/products.csv")
# 2.数据基本处理
# 2.1.合并表格
# 2.2.交叉表合并
# 2.3.数据截取
# 3.特征工程——PCA
# 4.机器学习——K-means
# 5.模型评估——平均轮廓系数
