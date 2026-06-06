import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, learning_curve, cross_val_score
from xgboost import XGBRegressor, callback
from tqdm import tqdm
from sklearn.metrics import r2_score, make_scorer,mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid, cross_val_score
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
from scipy.interpolate import make_interp_spline
plt.rcParams['font.family'] = 'Palatino Linotype'
# ------------------ 参数 ------------------
cv = KFold(n_splits=5, shuffle=True, random_state=123)

# 加载数据
file_path = 'table/Grid-Based_Feature-Engineered_data/fishnet_30x30_processed_feature_engineered.csv'
df = pd.read_csv(file_path)
X = df.iloc[1:, 0:-1]
y = df.iloc[1:, -1].values

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=108)

# 训练模型
model = XGBRegressor(
    learning_rate=0.01194543620793604,
    max_depth=11,
    max_leaves=256,
    grow_policy='depthwise',
    colsample_bytree=0.9784466574215168,
    colsample_bylevel=0.6192663461731014,
    colsample_bynode=0.8189964711585255,
    subsample=0.6489006448874424,
    min_child_weight=3,
    gamma=0.28620923204365845,
    reg_alpha=1.0476187038195663,
    reg_lambda=0.0004402425307719034,
    objective='reg:squarederror',
    tree_method='hist',
    seed=667,
    n_estimators=1000
)

model.fit(
    Xtrain, Ytrain,
    eval_set=[(Xtest, Ytest)],
    verbose=True
)

# 保存模型为pkl
model_path = r"E:\Desktop\5Gsimulation_XGboost\pythonProject\xgb_optuna_model_processed_feature_engineered.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")





# ------------------ optuna调参 ------------------
class TQDMCallback:
    def __init__(self, total):
        self.pbar = tqdm(total=total)

    def __call__(self, study, trial):
        self.pbar.update(1)
        self.pbar.set_postfix({"Best RMSE": f"{study.best_value:.4f}"})

    def close(self):
        self.pbar.close()

def xgb_optuna_tuner_save_model(Xtrain, Ytrain, model_path, n_trials=100, seed=667):
    """
    使用 Optuna 优化 XGBoost (原生API) 回归模型参数，支持早停与进度条，保存模型。

    参数：
        Xtrain, Ytrain: 训练数据 (numpy数组或DataFrame)
        model_path: 模型保存路径，如 'model.pkl'
        n_trials: 调参次数
        seed: 随机种子

    返回：
        best_model: 最佳xgboost.Booster模型
        best_params: 最佳参数字典
        best_score: 最佳验证集RMSE
    """

    def objective(trial):
        params = {
            'eta': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 12),
            'max_leaves': trial.suggest_categorical('max_leaves', [31, 64, 128, 256]),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'seed': seed,
        }

        # 划分训练集和验证集 (20%)
        X_tr, X_val, y_tr, y_val = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=seed)

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, 'train'), (dvalid, 'eval')]

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=30,
            verbose_eval=False
        )

        return model.best_score  # 验证集最优rmse

    print("📦 开始 Optuna 调参（带早停和进度条）...")

    progress_bar = TQDMCallback(total=n_trials)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[progress_bar])
    finally:
        progress_bar.close()

    best_params = study.best_trial.params
    best_params.update({
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'seed': seed,
    })

    # 用最优参数训练完整模型（无验证集）
    dtrain_full = xgb.DMatrix(Xtrain, label=Ytrain)
    best_model = xgb.train(
        best_params,
        dtrain_full,
        num_boost_round=1000,
        verbose_eval=False
    )

    # 保存模型
    best_model.save_model(model_path)
    print(f"✅ 模型已保存至: {model_path}")

    return best_model, best_params, study.best_value
# ------------------ shap分析 ------------------


import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

def shap_selected_features(
    model_path,
    csv_path,
    selected_features=None,  # 默认None表示用全部特征
    model_format='pkl'
):
    # 加载数据
    df = pd.read_csv(csv_path)
    X_full = df.iloc[1:, 0:-1].copy()  # 跳过第一行和最后一列

    # 如果没有指定selected_features，就用所有特征
    if selected_features is None:
        X_selected = X_full
    else:
        X_selected = X_full[selected_features]

    # 加载模型
    if model_format == 'pkl':
        model = joblib.load(model_path)
    else:
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(model_path)

    # 计算 SHAP 值
    print("🔍 正在计算 SHAP 值...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_selected)

    # 绘制 SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values.values, X_selected, show=True)




# ------------------ 分析各个特征对结果的影响 ------------------

def plot_pdp_selected_features(
    model_path,
    csv_path,
    selected_features=[
        'DEM_center',
        'Power_to_Dist_ratio',
        'True_3D_Dist',
        'ALT_M_',
        'Match_Dist',
        'DEM_center',
        'log_True_3D_Dist',
        'Building_Coverage',
        'SPEED_M_s_',
        'Match_Angle_cos',
        'NDVI_center',
        'Weighted_height',
        'Match_Angle_sin',
        'Match_Dist_bin'
    ],
    model_format='pkl',
    grid_resolution=1000,
    figsize=(6, 4),
    smooth=True
):
    """
    稳定、兼容所有 sklearn 版本的 PDP 绘图函数（仅提取 lines_ 数据）
    """
    # 加载模型
    if model_format == 'pkl':
        model = joblib.load(model_path)
    else:
        raise ValueError("仅支持 pkl 格式模型")

    # 读取数据
    df = pd.read_csv(csv_path)
    X = df.iloc[1:, 0:-1].copy()

    # 保留实际存在的特征
    available_features = [f for f in selected_features if f in X.columns]

    print(f"\n📊 绘制 PDP 图（共 {len(available_features)} 个特征）:")
    for i, feat in enumerate(available_features, 1):
        print(f"{i}. {feat}")

        # 临时图用于生成 lines_ 数据（我们不直接显示它）
        fig, ax = plt.subplots(figsize=figsize)
        disp = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=[feat],
            kind="average",
            grid_resolution=grid_resolution,
            ax=ax
        )

        # 提取 lines_ 中的数据
        x_vals, y_vals = disp.lines_[0][0].get_data()

        plt.close(fig)  # 关闭中间图，避免图像叠加或窗口过多

        # 手动绘图
        plt.figure(figsize=figsize, dpi=300)
        if smooth and len(x_vals) > 10:
            xnew = np.linspace(np.min(x_vals), np.max(x_vals), 300)
            ynew = make_interp_spline(x_vals, y_vals, k=3)(xnew)
            plt.plot(xnew, ynew, color='blue')
        else:
            plt.plot(x_vals, y_vals, color='blue')

        plt.title(f"PDP of {feat}")
        plt.xlabel(feat)
        plt.ylabel("RSRP Prediction")
        plt.grid(True)
        plt.tight_layout()
        # 保存图片，路径和文件名可根据需要修改
        save_path = fr"E:\Desktop\Supplementary picture\PDP_{feat}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()




# ------------------ 调参入口 ------------------
# best_model = grid_search_xgboost(param_grid, Xtrain, Ytrain, cv)
# print("测试集评估 RMSE:", np.sqrt(mean_squared_error(Ytest, best_model.predict(Xtest))))
# best_n_estimators = tune_n_estimators(Xtrain, Ytrain, cv)
# print("建议使用 n_estimators =", best_n_estimators)

#best_params, results_df = tune_maxdepth_minchildweight(Xtrain, Ytrain, cv)

#best_params, results_df = tune_subsample_colsample(Xtrain, Ytrain, cv=cv)

# #optuna调参
# model_path = r"E:\Desktop\5Gsimulation_XGboost\pythonProject\xgb_optuna_model.json"
# best_model, best_params, best_rmse = xgb_optuna_tuner_save_model(Xtrain, Ytrain, model_path=model_path, n_trials=200)
# print("调参完成，最佳参数：", best_params)
# print(f"最佳验证集RMSE: {best_rmse:.4f}")
# SHAP使用示例
model_path = r"E:\Desktop\5Gsimulation_XGboost\pythonProject\xgb_optuna_model_processed_feature_engineered.pkl"
csv_path = r"table/fishnet_30x30_processed_feature_engineered.csv"
# #
shap_selected_features(model_path, csv_path, selected_features=None)
# plot_pdp_selected_features(model_path, csv_path)
