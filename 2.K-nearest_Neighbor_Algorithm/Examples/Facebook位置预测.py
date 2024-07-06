# 0.导包
# 0.1 pandas导入数据集
import pandas as pd
# 0.2 数据集分割
from sklearn.model_selection import train_test_split, GridSearchCV
# 0.3 数据标准化处理
from sklearn.preprocessing import StandardScaler
# 0.4 机器学习 KNN+CV
from sklearn.neighbors import KNeighborsClassifier

# 1.读取数据
original_train_data = pd.read_csv("../../Data/facebook-vpredicting-check-ins/train.csv")

# 2.基本数据处理
# 2.1 最小化数据集
train_data = original_train_data.query("x>2.0 & x < 2.5 & y>2.0 & y<2.5")
# 2.2 选择时间特征
time = pd.to_datetime(train_data["time"], unit='s')     # unit声明时间单位秒s
time = pd.DatetimeIndex(time)
train_data["day"] = time.day
train_data["hour"] = time.hour
train_data["weekday"] = time.weekday
# 2.3 去除签到较少的地方
place_count = train_data.groupby("place_id").count()
place_count = place_count[place_count["row_id"] > 1]
train_data = train_data[train_data["place_id"].isin(place_count.index)]
# 2.4 选定特征值和目标值
x = train_data[["x", "y", "accuracy", "day", "hour", "weekday"]]
y = train_data["place_id"]
# 2.5 分割数据集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=20
)

# 3.特征工程——特征预处理（标准化）
# 3.1 实例化一个转换器
transfer = StandardScaler()
# 3.2 调用fit_transfer
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 4.机器学习 -- KNN+CV
# 4.1 实例化一个估计器
estimator = KNeighborsClassifier()
# 4.2 调用GridSearchCV
param_grid = {"n_neighbors": [1, 3, 5]}
estimator = GridSearchCV(estimator, param_grid, cv=0)
# 4.3 模型训练
estimator.fit(x_train, y_train)

# 5.模型评估
# 5.1 基本评估方式
score = estimator.score(x_test, y_test)
print("最后预期的准确率为：", score)

y_predict = estimator.predict(x_test)
print("最后的预测值为：\n", y_predict)
print("预测值和真实值的对比情况：\n", y_predict == y_test)

# 5.2 使用交叉验证后的评估方式
print("在交叉验证过程中最好的结果：\n", estimator.best_score_)
print("在交叉验证中最好的参数模型：\n", estimator.best_estimator_)
print("每次交叉验证后的验证集准确率结果和训练集准确率结果：\n", estimator.cv_results_)
