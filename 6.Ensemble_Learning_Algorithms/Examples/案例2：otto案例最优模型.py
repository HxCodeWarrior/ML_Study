import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

# 1.获取数据
original_data = pd.read_csv("../../Data/otto-group-product-classification-challenge/train.csv")

# 2.数据基本处理
# 2.1.确定特征值、目标值
x = original_data.drop(["id", "target"], axis=1)
y = original_data["target"]
# 2.2.随机欠采样数据
rus = RandomUnderSampler(random_state=0)
x_resampled, y_resampled = rus.fit_resample(x, y)
# 2.3.标签转换为数字
le = LabelEncoder()
y_resampled = le.fit_transform(y_resampled)
# 2.4.数据集切分
x_train, x_test, y_train, y_test = train_test_split(
    x_resampled,
    y_resampled,
    random_state=22,
    train_size=0.8
)

# 3.最优模型训练
rf3 = RandomForestClassifier(
    n_estimators=175,
    max_features=13,
    max_depth=35,
    min_samples_leaf=1,
    random_state=40,
    n_jobs=-1,
    oob_score=True
)
rf3.fit(x_train, y_train)

# 4.模型评估
rf3_score = rf3.score(x_test, y_test)
rf3_oob_score = rf3.oob_score_
y_pre_proba = rf3.predict_proba(x_test)
# one_hot = OneHotEncoder()
# y_true = one_hot.fit_transform(y_test.reshape(-1, 1)).toarray()
# y_pred = one_hot.fit_transform(y_pre_proba.reshape(-1, 1)).toarray()
rf3_log_loss = log_loss(y_test, y_pre_proba, normalize=True)
print("最优模型精确率：", rf3_score)
print("最优模型：", rf3_oob_score)
print("最优模型logloss：", rf3_log_loss)

# 5.模型测试
test = pd.read_csv("../../Data/otto-group-product-classification-challenge/test.csv")
test_data = test.drop(["id"], axis=1)
y_test_pred = rf3.predict_proba(test_data)
result_data = pd.DataFrame(y_test_pred, columns=[f"Class_{i}" for i in range(1, 10)])
result_data.insert(loc=0, column="id", value=test.id)
result_data.to_csv("Runs/test.csv")
print(result_data)

