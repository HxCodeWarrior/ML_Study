# 0.导入模块
# 导入鸢尾花数据集
from sklearn.datasets import load_iris
# 数据集分割
from sklearn.model_selection import train_test_split, GridSearchCV
# 数据标准化处理
from sklearn.preprocessing import StandardScaler
# k近邻算法
from sklearn.neighbors import KNeighborsClassifier

# 1.从sklearn中获取数据集
iris = load_iris()

# 数据集属性描述
print("数据集特征值：\n", iris.data)
print("数据集目标值：\n", iris["target"])
print("数据集特征值名字：\n", iris.feature_names)
print("数据集目标值名字：\n", iris.target_names)
print("数据集的描述：\n", iris.DESCR)

# 2.对数据集进行分割
"""
x_train:
x_test:
y_train:
y_test:
"""
x_train, x_test, y_train, y_test = train_test_split(
    iris.data,  #特征值
    iris.target,  #目标值
    test_size=0.2,
    random_state=20
)

# 3.特征工程——数据标准化
# 3.1实例化一个标准化对象
transfer = StandardScaler()
# 3.2数据标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4.机器学习--KNN
# 4.1实例化一个估计器类
estimator = KNeighborsClassifier(n_neighbors=5)
# 模型训练estimator.fit(x_train, y_train)
# 4.2 模型选择与调优--网格搜索和交叉验证
# 4.2.1 准备超参数
param_dict = {"n_neighbors": [1, 3, 5]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
# 4.3 fit数据进行训练
estimator.fit(x_train, y_train)

# 5.模型评估
# 方法1：对比真实值和预测值
y_pred = estimator.predict(x_test)
print("预测结果为：\n", y_pred)
print("对比真实值和预测值：\n", y_pred == y_test)
# 方法2：直接计算准确率
score = estimator.score(x_test, y_test)
print("准确率为：", score)

# 6.评估查看最终选择的结果和交叉验证的结果
print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
print("最好的参数模型：\n", estimator.best_estimator_)
print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)
