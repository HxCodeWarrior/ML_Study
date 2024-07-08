import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def linear_model():
    # 1.获取数据
    boston_data = pd.read_csv("../../Data/boston_house_prices.csv")
    print(boston_data)

    # 2.数据基本处理-数据集划分
    x = boston_data.drop("MEDV", axis=1)  # 删除目标值列，保留特征值
    y = boston_data["MEDV"]  # 提取目标值列
    x_train, x_test, y_train, y_test = train_test_split(
        x,  # 特征值
        y,  # 目标值
        test_size=0.2  # 测试集占比20%
    )

    # 3.特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4.机器学习-线性回归（正规方程）
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 5.1.获取系数等值
    print("模型的偏置：\n", estimator.intercept_)
    print("模型的系数：\n", estimator.coef_)

    # 5.2.评价
    y_pred = estimator.predict(x_test)
    ret = mean_squared_error(y_test, y_pred)
    print("预测值：\n", y_pred)
    print("均方误差: \n", ret)


linear_model()
