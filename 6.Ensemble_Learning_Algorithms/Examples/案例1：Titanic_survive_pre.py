import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

# 读取数据
data = pd.read_csv("../../Data/titanic/train.csv")

# 数据基本处理
# 确定特征值、目标值
x = data[["Pclass", "Age", "Sex"]]
y = data["Survived"]
# 处理缺失值
x["Age"].fillna(value=x["Age"].mean(), inplace=True)
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, train_size=0.75)

# 特征工程——字典特征提取
x_train = x_train.to_dict(orient="records")
x_test = x_test.to_dict(orient="records")
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 机器学习——随机森林
RandomForest = RandomForestClassifier()
# 定义超参数列表
param = {"n_estimators": [120, 200, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
# 使用GridSearchCV进行网格搜索
gc = GridSearchCV(RandomForest, param_grid=param, cv=2)
gc.fit(x_train, y_train)

# 模型评估
print("随机森林预测的准确率为：", gc.score(x_test, y_test))
