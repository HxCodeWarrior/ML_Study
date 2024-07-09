# 0.导包
import numpy as np
import pandas as pd  # 获取数据集，处理数据
from sklearn.model_selection import train_test_split  # 数据集分割
from sklearn.preprocessing import StandardScaler  # 特征工程--标准化
from sklearn.linear_model import LogisticRegression  # 机器学习--逻辑回归
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score  # 模型评估

# 1.获取数据集
columns_names = ["Sample_code_number", "Clump_thickness", "Uniformity_of_cell_size",
                 "Uniformity_of_cell_shape", "Marginal_adhesion", "Single_epithelial_cell_size",
                 "Bare_nuclei", "Bland_chromatin", "Normal_nucleoli", "Mitoses", "Class"
                 ]
breast_cancer_wisconsin = pd.read_csv(
    "../../Data/Breast_Cancer_Wisconsin/breast-cancer-wisconsin.data",
    names=columns_names
)

# 2.数据基本处理
# 2.1.处理缺失值
data = breast_cancer_wisconsin.replace(to_replace="?", value=np.nan)
data = data.dropna()
# 2.2.确定特征值、目标值
x = data.iloc[:, 1:10]
y = data["Class"]
# 2.3.数据集分割
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25
)

# 3.特征工程--标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 4.机器学习--逻辑回归
estimator = LogisticRegression()
estimator.fit(x_train, y_train)

# 5.模型评估
y_predict = estimator.predict(x_test)
ret1 = mean_squared_error(y_test, y_predict)
ret2 = classification_report(y_test, y_predict, labels=[2, 4], target_names=("良性", "恶性"))
y_test = np.where(y_test > 3, 1, 0)
ret3 = roc_auc_score(y_test, y_predict)
print("预测值：\n", y_predict)
print("准确率：", estimator.score(x_test, y_test))
print("模型均方误差：", ret1)
print("召回率评价：\n", ret2)
print("模型AUC指标：", ret3)
