# 0.导包
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

# 1.获取数据集
boston_data = pd.read_csv("../../Data/boston_house_prices.csv")

# 2.数据集划分
x = boston_data.drop("MEDV", axis=1)
y = boston_data["MEDV"]
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2
)

# 3.数据基本处理--标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_teat = transfer.fit_transform(x_test)

# 4.机器学习--线性回归（岭回归）
estimator = RidgeCV(alphas=tuple(i for i in range(1, 11)))
# 4.1.训练模型
estimator.fit(x_train, y_train)
# 4.2.保存模型
# joblib.dump(estimator, "predict_house_prices.pki")
# 4.3.加载模型
estimator = joblib.load("predict_house_prices.pki")

# 5.模型你评估
# 5.1.获取模型系数
y_predict = estimator.predict(x_test)
print("预测值为：\n", y_predict)
print("模型偏置为：\n", estimator.intercept_)
print("模型系数为：\n", estimator.coef_)

# 5.2.均方差评价
ret = mean_squared_error(y_test, y_predict)
print("模型均方误差为：\n", ret)
