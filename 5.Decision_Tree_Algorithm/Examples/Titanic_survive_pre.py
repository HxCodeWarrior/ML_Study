# 0.导包
import pandas as pd  # 获取数据
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.feature_extraction import DictVectorizer  # 字典特征提取
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # 机器学习-决策树
from sklearn.metrics import mean_squared_error  # 模型评估

# 1.获取数据
train_data = pd.read_csv("../../Data/titanic/train.csv")

# 2.数据基本处理
# 2.1.确定特征值、目标值
x = train_data[["Pclass", "Sex", "Age"]]
y = train_data["Survived"]
# 2.2.缺失值处理
x["Age"].fillna(value=train_data["Age"].mean(), inplace=True)
# 2.3.数据集划分
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=22,
    train_size=0.75
)
# 3.特征工程——字典特征提取
x_train = x_train.to_dict(orient="records")
x_test = x_test.to_dict(orient="records")
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 4.机器学习——决策树
estimator = DecisionTreeClassifier()
estimator.fit(x_train, y_train)

# 5.模型评估
y_predict = estimator.predict(x_test)
score = estimator.score(x_test, y_test)
mean_squared = mean_squared_error(y_test, y_predict)
print("预测值：\n", y_predict)
print("准确率：\n", score)
print("模型召回率：\n", mean_squared)

# 6.决策树可视化
try:
    export_graphviz(estimator, out_file="Titanic_tree.dot", feature_names=["Age", "Pclass", "女性", "男性"])
    print("决策树Titanic_tree.dot保存成功！")
except:
    print("保存失败！")
