from sklearn.datasets import load_iris

# 获取鸢尾花数据集(小数据集)
iris = load_iris()

# 数据集属性描述
print("数据集特征值：", iris.data)
print("数据集目标值：", iris["target"])
print("数据集特征值名字：", iris.feature_names)
print("数据集目标值名字：", iris.target_names)
print("数据集的描述：", iris.DESCR)
