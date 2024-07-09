# 0.导包
import pandas as pd  # 获取数据集
from sklearn.model_selection import train_test_split  # 分割数据集
from sklearn.preprocessing import StandardScaler  # 特征工程-标准化处理
from sklearn.linear_model import Ridge, RidgeCV  # 机器学习-岭回归
from sklearn.metrics import mean_squared_error  # 模型评估

# 1.获取数据集
boston_data = pd.read_csv("../../Data/boston_house_prices.csv")

# 2.数据基本处理--分割数据集
x = boston_data.drop("MEDV", axis=1)  # 获取特征值
y = boston_data["MEDV"]  # 获取目标值
x_train, x_test, y_train, y_test = train_test_split(
    x,  # 特征值
    y,  # 目标值
    test_size=0.2  # 测试集占比20%
)

# 3.特征工程--标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 4.机器学习--线性回归（岭回归）
estimator = RidgeCV(alphas=(0.001, 0.1, 1, 10, 100))
# estimator = Ridge(alpha=1)
estimator.fit(x_train, y_train)

# 5.模型评估
print("这个模型的偏置：\n", estimator.intercept_)
print("这个模型的系数：\n", estimator.coef_)
# 5.1.预测值
y_pre = estimator.predict(x_test)
print("预测值为：\n", y_pre)
# 5.1.均方误差
ret = mean_squared_error(y_test, y_pre)
print("模型均方误差为：\n", ret)
