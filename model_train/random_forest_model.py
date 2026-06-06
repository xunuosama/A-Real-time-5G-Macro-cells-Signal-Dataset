import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# 1. 加载数据
# ===============================
file_path = '../table/Grid-Based_Feature-Engineered_data/fishnet_30x30_processed_feature_engineered.csv'
df = pd.read_csv(file_path)

# 和你 XGBoost 保持一致
X = df.iloc[1:, 0:-1]
y = df.iloc[1:, -1].values

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, y, test_size=0.2, random_state=108
)

# ===============================
# 2. 训练 Random Forest
# ===============================
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=667,
    n_jobs=-1
)

rf_model.fit(Xtrain, Ytrain)

# ===============================
# 3. 测试集预测
# ===============================
Ypred = rf_model.predict(Xtest)

mae = mean_absolute_error(Ytest, Ypred)
rmse = np.sqrt(mean_squared_error(Ytest, Ypred))
r2 = r2_score(Ytest, Ypred)

print("Random Forest model")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R2: {r2:.6f}")

# ===============================
# 4. 保存模型
# ===============================
model_path = r"E:\Desktop\5Gsimulation_XGboost\pythonProject\rf_model_processed_feature_engineered.pkl"

os.makedirs(os.path.dirname(model_path), exist_ok=True)

joblib.dump(rf_model, model_path)