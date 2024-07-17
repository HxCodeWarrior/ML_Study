# 0.导包
import os
import numpy as np  # 模型调优
import pandas as pd  # 读取数据
import matplotlib.pyplot as plt  # 数据可视化
import seaborn as sns  # 采样结果可视化
from imblearn.under_sampling import RandomUnderSampler  # 随机欠采样
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.preprocessing import OneHotEncoder, LabelEncoder  # 多分类转换one-hot，标签转换为数字
from sklearn.ensemble import RandomForestClassifier  # 模型训练
from sklearn.metrics import log_loss


def data_view(detected_values):
    sns.countplot(detected_values)
    plt.show()


def data_process(eigen_value, target_value):
    """
    大量数据的处理（随机欠采样）
    :param eigen_value: 特征值
    :param target_value: 目标值
    :return:x_train_data, x_test_data, y_train_data, y_test_data
    """
    # 1.截取数据集--随机欠采样
    rus = RandomUnderSampler(random_state=0)
    x_resampled, y_resampled = rus.fit_resample(eigen_value, target_value)
    print("随机欠采样数据结果：\n"
          f"【x】:\n{x_resampled},"
          f"【y】：\n{y_resampled}\n"
          "采样后数据集详情:\n"
          f"【x】:\n{x_resampled.describe()}\n"
          f"【y】:\n{x_resampled.describe()}\n")

    # 2.标签转换为数字
    le = LabelEncoder()
    y_resampled = le.fit_transform(y_resampled)
    print(f"类别特征转换后结果：\n{y_resampled}")

    # 2.4.切分数据集
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_resampled, y_resampled, train_size=0.8)

    return x_train_data, x_test_data, y_train_data, y_test_data


def model_adjust(value_range: list, parameter_name: str, adjust_data: list,
                 random_state=0):
    """
    function:模型调优
    :param value_range:
    :param parameter_name:
    :param adjust_data:
    :param random_state:
    :return:
    """
    # 确定调优参数的取值范围
    turned_parameters_data = range(value_range[0], value_range[1], value_range[2])
    # 创建添加accuracy的一个numpy
    accuracy_t_data = np.zeros(len(turned_parameters))
    # 创建添加error的一个numpy
    error_t_data = np.zeros(len(turned_parameters))
    if parameter_name == "n_estimators":
        for i, one_parameter in enumerate(turned_parameters_data):
            rf2 = RandomForestClassifier(
                n_estimators=one_parameter,
                max_depth=adjust_data[1],
                max_features=adjust_data[2],
                min_samples_leaf=adjust_data[3],
                oob_score=True,
                random_state=random_state,
                n_jobs=-1
            )
            # 模型训练
            rf2.fit(x_train, y_train)
            # 输出accuracy
            accuracy_t[i] = rf2.oob_score_
            # 输出log_loss
            y_pre = rf2.predict_proba(x_test)
            error_t[i] = log_loss(y_test, y_pre, normalize=True)
            print(f"模型调优{parameter_name}:", error_t)
    elif parameter_name == "max_depth":
        for j, one_parameter in enumerate(turned_parameters_data):
            rf2 = RandomForestClassifier(
                n_estimators=adjust_data[0],
                max_depth=one_parameter,
                max_features=adjust_data[2],
                min_samples_leaf=adjust_data[3],
                oob_score=True,
                random_state=random_state,
                n_jobs=-1
            )
            # 模型训练
            rf2.fit(x_train, y_train)
            # 输出accuracy
            accuracy_t[j] = rf2.oob_score_
            # 输出log_loss
            y_pre = rf2.predict_proba(x_test)
            error_t[j] = log_loss(y_test, y_pre, normalize=True)
            print(f"模型调优{parameter_name}:", error_t)
    elif parameter_name == "max_features":
        for k, one_parameter in enumerate(turned_parameters_data):
            rf2 = RandomForestClassifier(
                n_estimators=adjust_data[0],
                max_depth=adjust_data[1],
                max_features=one_parameter,
                min_samples_leaf=adjust_data[3],
                oob_score=True,
                random_state=random_state,
                n_jobs=-1
            )
            # 模型训练
            rf2.fit(x_train, y_train)
            # 输出accuracy
            accuracy_t[k] = rf2.oob_score_
            # 输出log_loss
            y_pre = rf2.predict_proba(x_test)
            error_t[k] = log_loss(y_test, y_pre, normalize=True)
            print(f"模型调优{parameter_name}:", error_t)
    elif parameter_name == "max_samples_leaf":
        for l, one_parameter in enumerate(turned_parameters_data):
            rf2 = RandomForestClassifier(
                n_estimators=adjust_data[0],
                max_depth=adjust_data[1],
                max_features=adjust_data[2],
                min_samples_leaf=one_parameter,
                oob_score=True,
                random_state=random_state,
                n_jobs=-1
            )
            # 模型训练
            rf2.fit(x_train, y_train)
            # 输出accuracy
            accuracy_t[l] = rf2.oob_score_
            # 输出log_loss
            y_pre = rf2.predict_proba(x_test)
            error_t[l] = log_loss(y_test, y_pre, normalize=True)
            print(f"模型调优{parameter_name}:", error_t)
    else:
        print("调优参数不在可选范围内！")
    return turned_parameters_data, accuracy_t_data, error_t_data


def model_adjust_view(turned_parameters_data, error_t_data, accuracy_t_data, names: list, file_name=None,
                      save_path: str = "Runs/"):
    """
    function:模型调优过程的可视化
    :param turned_parameters_data:
    :param error_t_data:
    :param accuracy_t_data:
    :param names: 两个表格的横纵坐标轴名称
    :param file_name: 可选，文件名
    :param save_path: 可选，图片保存路径
    :return:
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4), dpi=100)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(turned_parameters_data, error_t_data)
    ax1.set_xlabel(names[0])
    ax1.set_ylabel(names[1])
    ax1.grid(True)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(turned_parameters_data, accuracy_t_data)
    ax2.set_xlabel(names[2])
    ax2.set_ylabel(names[3])
    ax2.grid(True)

    # 如果未提供文件名，则使用系统自动生成的名称
    if file_name is None:
        file_name = 'plot_{}.png'.format(np.random.randint(0, 1000))

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存图像到指定位置
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path)
    print('Image saved at:', file_path)

    # 检查文件是否已成功保存
    if os.path.exists(file_path):
        print("File saved successfully at", file_path)
    else:
        print("Error: File not saved at", file_path)


def model_evaluation(x_test_data, y_test_data):
    """
    function:模型评估
    :param x_test_data:
    :param y_test_data:
    :return: None
    """
    # 1.基本评估：预测值，准确率
    y_pre = rf.predict(x_test_data)
    score = rf.score(x_test_data, y_test_data)
    # data_view(y_pre)
    print("预测值：\n", y_pre)
    print("精确率：\n", score)

    # 5.2.log_loss模型评估
    # one_hot编码转换
    one_hot = OneHotEncoder()
    y_test_one_hot = one_hot.fit_transform(y_test.reshape(-1, 1))  # type:'scipy.sparse._csr.csr_matrix'
    y_pre_one_hot = one_hot.fit_transform(y_pre.reshape(-1, 1))

    # 将压缩稀疏行矩阵转换为一个普通的数组
    y_true = y_test_one_hot.toarray()
    y_pred = y_pre_one_hot.toarray()

    original_log_loss = log_loss(y_true, y_pred, normalize=True)
    print("log_loss: \n", original_log_loss)


if __name__ == '__main__':
    # 获取数据
    original_data = pd.read_csv("../../Data/otto-group-product-classification-challenge/train.csv")

    # 数据基本处理
    # 确定特征值、目标值
    x = original_data.drop(["id", "target"], axis=1)
    y = original_data["target"]
    # 数据处理
    x_train, x_test, y_train, y_test = data_process(x, y)

    # 机器学习--随机森林
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    # 模型评估
    model_evaluation(x_test, y_test)

    # 模型调优:n_estimator,max_feature,max_depth, min_samples_leaf
    # 1.n_estimator调优
    turned_parameters, accuracy_t, error_t = model_adjust([10, 200, 5], parameter_name="n_estimators", adjust_data=[10, 10, 10, 10])
    # 模型调优过程可视化
    model_adjust_view(turned_parameters, error_t, accuracy_t, ["n_estimators", "error_t", "n_estimators", "accuracy_t"],
                      file_name="n_estimator")

    # 2.max_feature调优
    turned_parameters, accuracy_t, error_t = model_adjust([10, 50, 2], parameter_name="max_features", adjust_data=[175, 10, 10, 10])
    # 模型调优过程可视化
    model_adjust_view(turned_parameters, error_t, accuracy_t, ["max_features", "error_t", "max_features", "accuracy_t"],
                      file_name="max_features")

    # 3.max_depth调优
    turned_parameters, accuracy_t, error_t = model_adjust([10, 100, 5], parameter_name="max_depth", adjust_data=[175, 13, 10, 10])
    # 模型调优过程可视化
    model_adjust_view(turned_parameters, error_t, accuracy_t, ["max_depth", "error_t", "max_depth", "accuracy_t"],
                      file_name="max_depth")

    # 4.max_samples_leaf调优
    turned_parameters, accuracy_t, error_t = model_adjust([1, 10, 1], parameter_name="max_samples_leaf",
                 adjust_data=[175, 13, 35, 10])
    # 模型调优过程可视化
    model_adjust_view(turned_parameters, error_t, accuracy_t,
                      ["max_samples_leaf", "error_t", "max_samples_leaf", "accuracy_t"],
                      file_name="max_samples_leaf")

"""
最优模型参数：
n_estimator:175
max_features:13
max_depth:35
max_samples_leaf:1
"""
