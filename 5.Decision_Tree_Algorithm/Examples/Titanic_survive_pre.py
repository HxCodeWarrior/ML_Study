# 0.导包
import pandas as pd     # 获取数据
from sklearn.model_selection import train_test_split     # 数据集划分
from sklearn.feature_extraction import DictVectorizer    # 字典特征提取
from sklearn.tree import DecisionTreeClassifier, export_graphviz    # 机器学习-决策树

# 1.获取数据
train_data = pd.read_csv("../../Data/titanic/train.csv")

# 2.数据基本处理
# 2.1.确定特征值、目标值
x = train_data[["Plass","Sex","Age"]]
y = train_data["Survived"]
# 2.2.缺失值处理

# 2.3.数据集划分
# 3.特征工程——字典特征提取
# 4.机器学习——决策树
# 5.模型评估
