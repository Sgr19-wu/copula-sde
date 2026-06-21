# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:38:36 2025

@author: qiannnn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import shap
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与预处理
df = pd.read_csv(r"D:\降水与径流量\data\data1.csv")
df["date"] = pd.to_datetime(df["date"])
print(f"原始数据量: {len(df)}")

# 2. 自适应特征生成
def generate_adaptive_features(df, max_lag=30):
    df_feat = df.copy()
    
    # 1. 基础时间特征
    df_feat["month"] = df_feat["date"].dt.month
    df_feat["dayofyear"] = df_feat["date"].dt.dayofyear
    df_feat["sin_month"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["cos_month"] = np.cos(2 * np.pi * df_feat["month"] / 12)
    df_feat["sin_doy"] = np.sin(2 * np.pi * df_feat["dayofyear"] / 365)
    df_feat["cos_doy"] = np.cos(2 * np.pi * df_feat["dayofyear"] / 365)
    
    # 2. 自适应滞后特征
    base_vars = ['precipitation', 'temperature', 'swe']
    lag_features = []
    for var in base_vars:
        for lag in range(1, max_lag + 1):
            col_name = f"{var}_lag{lag}"
            df_feat[col_name] = df_feat[var].shift(lag)
            lag_features.append(col_name)
    print(f"生成滞后特征: {len(base_vars)} 变量 × {max_lag} 天 = {len(lag_features)} 个")
    
    # 3. 移动统计特征
    windows = [3, 7]
    stat_features = []
    for var in base_vars:
        for window in windows:
            col_ma = f"{var}_ma{window}"
            df_feat[col_ma] = df_feat[var].rolling(window=window).mean()
            stat_features.append(col_ma)
    print(f"生成移动统计特征: {len(stat_features)} 个")
    
    # 4. 累积特征
    sum_features = []
    for window in [3, 7, 14]:
        col_sum = f"precip_sum{window}"
        df_feat[col_sum] = df_feat["precipitation"].rolling(window=window).sum()
        sum_features.append(col_sum)
    print(f"生成累积特征: {len(sum_features)} 个")
    
    # 5. 交互特征
    inter_features = [
        "temp_precip", "swe_temp", "swe_precip"
    ]
    df_feat["temp_precip"] = df_feat["temperature"] * df_feat["precipitation"]
    df_feat["swe_temp"] = df_feat["swe"] * df_feat["temperature"]
    df_feat["swe_precip"] = df_feat["swe"] * df_feat["precipitation"]
    print(f"生成交互特征: {len(inter_features)} 个")
    
    # 6. 特征列表汇总
    all_features = (
        base_vars +
        lag_features +
        stat_features +
        sum_features +
        inter_features +
        ["sin_month", "cos_month", "sin_doy", "cos_doy"]
    )
    
    # 删除缺失值
    df_feat = df_feat.dropna().reset_index(drop=True)
    print(f"  总特征数: {len(all_features)}")
    print(f"  有效样本数: {len(df_feat)}")
    
    return df_feat, all_features

# 生成自适应特征集
df_full, X_cols_all = generate_adaptive_features(df, max_lag=30)

# 3. 数据准备 - 不使用log变换
y_up = df_full["discharge_upstream"].values
y_down = df_full["discharge_downstream"].values

# 4. 时间序列划分
split_idx = int(len(df_full) * 0.8)
X_train, X_test = df_full[X_cols_all].values[:split_idx], df_full[X_cols_all].values[split_idx:]

y_train_up, y_test_up = y_up[:split_idx], y_up[split_idx:]
y_train_down, y_test_down = y_down[:split_idx], y_down[split_idx:]

print(f"\n数据划分:")
print(f"  训练集: {len(X_train)} 样本 ({len(X_train)/len(df_full)*100:.1f}%)")
print(f"  测试集: {len(X_test)} 样本 ({len(X_test)/len(df_full)*100:.1f}%)")

# 5. 自适应滞后期选择器
def select_optimal_lags(model, feature_names, cum_importance_thresh=0.90, corr_thresh=0.85, min_features=8):
    # 1. 提取特征重要性并排序
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    
    # 2. 计算累积重要性，筛选核心特征
    feat_imp["Cum_Importance"] = feat_imp["Importance"].cumsum()
    # 先按累积重要性筛选
    core_feats_df = feat_imp[feat_imp["Cum_Importance"] <= cum_importance_thresh]
    core_feats = core_feats_df["Feature"].tolist()
    # 如果特征太少，强制保留前min_features个
    if len(core_feats) < min_features:
        core_feats_df = feat_imp.head(min_features)
        core_feats = core_feats_df["Feature"].tolist()
        print(f"  ⚠️  特征不足{min_features}个，强制保留前{min_features}个核心特征")

    # 3. 相关性过滤，剔除高度冗余特征（但确保过滤后至少有min_features//2个特征）
    if len(core_feats) > 1:
        corr_matrix = df_full[core_feats].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
        # 防止过滤后特征太少
        if len(core_feats) - len(to_drop) < min_features//2:
            to_drop = to_drop[:len(core_feats) - (min_features//2)]  # 只剔除部分冗余特征
        core_feats = [f for f in core_feats if f not in to_drop]
        core_feats_df = core_feats_df[core_feats_df["Feature"].isin(core_feats)]
        print(f"  相关性过滤: 剔除了 {len(to_drop)} 个高度相关特征")

    # 4. 分析筛选后的滞后期分布
    lag_analysis = {}
    for var in ['precipitation', 'temperature', 'swe']:
        var_lags = [f for f in core_feats if f.startswith(f"{var}_lag")]
        if var_lags:
            lag_nums = [int(f.split('lag')[1]) for f in var_lags]
            lag_analysis[var] = {
                "selected_lags": lag_nums,
                "min_lag": min(lag_nums),
                "max_lag": max(lag_nums),
                "avg_lag": np.mean(lag_nums),
                "count": len(lag_nums)
            }
    
    print(f"\n最优滞后期分析:")
    for var, info in lag_analysis.items():
        print(f"  {var}: 选中滞后{info['count']}个")
    
    print(f"\n特征筛选结果:")
    print(f"  原始特征数: {len(feature_names)}")
    print(f"  筛选后特征数: {len(core_feats)}")
    print(f"  精简率: {(1 - len(core_feats)/len(feature_names))*100:.1f}%")
    
    return core_feats, core_feats_df

# 6. 主训练函数（无log变换版本）
def train_adaptive_gbm_no_log(X_train, y_train, X_test, y_test, X_full, y_full, 
                             feature_names, label, max_lag=30):
    
    # 动态超参数配置
    if label == "Upstream":
        print("\n上游流域:")
        # 上游使用Huber损失，对异常值更稳健
        base_loss = 'huber'
        base_alpha = 0.95
        cum_importance_thresh = 0.92
        min_features = 10
        base_n_iter_no_change = 30
        base_validation_fraction = 0.2
        
        # 上游超参数网格
        param_grid = {
            'learning_rate': [0.006, 0.007, 0.008], 
            'max_depth': [5, 6],
            'min_samples_split': [8, 10, 12],
            'min_samples_leaf': [12, 13, 14], 
            'subsample': [0.75, 0.8, 0.85],
            'max_features': [0.55, 0.6, 0.65]
        }
    else:  # Downstream
        print("\n下游流域:")
        # 下游使用Huber损失
        base_loss = 'huber'
        base_alpha = 0.95
        cum_importance_thresh = 0.92
        min_features = 12
        base_n_iter_no_change = 50
        base_validation_fraction = 0.1
        
        # 下游超参数网格
        param_grid = {
            'learning_rate': [0.02, 0.03],
            'max_depth': [3, 4],
            'min_samples_split': [20, 25],
            'min_samples_leaf': [10, 15], 
            'subsample': [0.75, 0.8, 0.85], 
            'max_features': [0.55, 0.6, 0.65]
        }

    # 步骤1: 训练基础模型
    base_model = GradientBoostingRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=4,
        loss=base_loss,
        alpha=base_alpha,
        random_state=42,
        n_iter_no_change=base_n_iter_no_change,
        validation_fraction=base_validation_fraction
    )
    base_model.fit(X_train, y_train)
    
    # 步骤2: 自适应筛选最优滞后期和核心特征
    core_feats, core_feat_imp = select_optimal_lags(
        model=base_model,
        feature_names=feature_names,
        cum_importance_thresh=cum_importance_thresh,
        corr_thresh=0.85,
        min_features=min_features  
    )
    core_feat_idx = [feature_names.index(f) for f in core_feats]
    X_train_core = X_train[:, core_feat_idx]
    X_test_core = X_test[:, core_feat_idx]
    X_full_core = X_full[:, core_feat_idx]
    
    # 步骤3: 随机搜索优化最终模型
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(
            n_estimators=1500,
            loss=base_loss,
            alpha=base_alpha,
            n_iter_no_change=base_n_iter_no_change,
            validation_fraction=base_validation_fraction,
            random_state=42
        ),
        param_distributions=param_grid,
        n_iter=30,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    random_search.fit(X_train_core, y_train)
    best_model = random_search.best_estimator_
    
    print(f"\n最优参数:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"早停后实际树数量: {best_model.n_estimators_}")
    print(f"CV最佳RMSE: {-random_search.best_score_:.2f}")
    
    # 保存模型
    model_save_dir = "./best_models_no_log"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    joblib.dump(best_model, os.path.join(model_save_dir, f"{label}_best_model.pkl"))

    # 步骤4: 模型评估（原始尺度）
    y_train_pred = best_model.predict(X_train_core)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    y_test_pred = best_model.predict(X_test_core)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / np.maximum(y_test, 1))) * 100  # 避免除零
    
    print("\n模型性能 (原始尺度):")
    print(f"  训练集: R²={train_r2:.4f}, RMSE={train_rmse:.2f}")
    print(f"  测试集: R²={test_r2:.4f}, RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, MAPE={test_mape:.2f}%")
    
    # 步骤5: 残差分析（原始尺度）
    y_full_pred = best_model.predict(X_full_core)
    residuals = y_full - y_full_pred
    
    print(f"\n残差统计 (原始尺度):")
    print(f"  均值: {residuals.mean():.2f}, 标准差: {residuals.std():.2f}")
    print(f"  偏度: {stats.skew(residuals):.4f}, 峰度: {stats.kurtosis(residuals):.4f}")
    
    # 步骤6: 可视化 
    fig = plt.figure(figsize=(22, 18))
    
    # 1. 预测vs实际（测试集）
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(y_test, y_test_pred, alpha=0.4, s=20, c='steelblue', edgecolors='navy', linewidth=0.5)
    lim = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    ax1.plot(lim, lim, 'r--', lw=2.5, label='完美预测线')
    ax1.set_xlabel('真实值', fontsize=12)
    ax1.set_ylabel('预测值', fontsize=12)
    ax1.set_title(f'{label} - 预测vs实际 (测试集)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'R²={test_r2:.4f}\nRMSE={test_rmse:.1f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. 残差分布
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(residuals, bins=80, alpha=0.7, color='lightcoral', density=True, edgecolor='darkred', linewidth=0.5)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2.5, label='零线')
    mu, std = residuals.mean(), residuals.std()
    x_norm = np.linspace(residuals.min(), residuals.max(), 200)
    ax2.plot(x_norm, stats.norm.pdf(x_norm, mu, std), 'b-', lw=2.5, label='正态拟合')
    ax2.set_xlabel('残差', fontsize=12)
    ax2.set_ylabel('密度', fontsize=12)
    ax2.set_title(f'{label} - 残差分布', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q图（正态性检验）
    ax3 = plt.subplot(3, 3, 3)
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.get_lines()[0].set_markersize(4)
    ax3.get_lines()[0].set_alpha(0.5)
    ax3.get_lines()[1].set_color('red')
    ax3.get_lines()[1].set_linewidth(2.5)
    ax3.set_title(f'{label} - Q-Q图', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 筛选后Top15特征重要性
    ax4 = plt.subplot(3, 3, 4)
    top15_imp = core_feat_imp.sort_values("Importance", ascending=False).head(15)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top15_imp)))
    ax4.barh(range(len(top15_imp)), top15_imp['Importance'], color=colors, edgecolor='black', linewidth=0.5)
    ax4.set_yticks(range(len(top15_imp)))
    ax4.set_yticklabels(top15_imp['Feature'], fontsize=10)
    ax4.set_xlabel('重要性', fontsize=12)
    ax4.set_title(f'{label} - Top15 核心特征重要性', fontsize=13, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. 残差vs预测值（异方差检验）
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(y_full_pred, residuals, alpha=0.3, s=10, c='coral', edgecolors='none')
    ax5.axhline(0, color='red', linestyle='--', linewidth=2.5)
    sort_idx = np.argsort(y_full_pred)
    y_sorted = y_full_pred[sort_idx]
    r_sorted = residuals[sort_idx]
    window = max(10, len(y_sorted) // 20)
    rolling_std = pd.Series(np.abs(r_sorted)).rolling(window=window, center=True).mean()
    ax5.plot(y_sorted, rolling_std, 'b-', linewidth=2.5, label='|残差|滚动均值')
    ax5.plot(y_sorted, -rolling_std, 'b-', linewidth=2.5)
    ax5.set_xlabel('预测值', fontsize=12)
    ax5.set_ylabel('残差', fontsize=12)
    ax5.set_title(f'{label} - 残差vs预测值', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 时间序列预测图
    ax6 = plt.subplot(3, 3, 6)
    dates = df_full["date"].iloc[split_idx:split_idx+len(y_test)]
    ax6.plot(dates, y_test, 'b-', label='真实值', linewidth=1.5, alpha=0.8)
    ax6.plot(dates, y_test_pred, 'r-', label='预测值', linewidth=1.5, alpha=0.8)
    ax6.set_xlabel('日期', fontsize=12)
    ax6.set_ylabel('径流量', fontsize=12)
    ax6.set_title(f'{label} - 时间序列预测', fontsize=13, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # 7. 滞后特征重要性热力图
    ax7 = plt.subplot(3, 3, 7)
    lag_feats_df = core_feat_imp[core_feat_imp['Feature'].str.contains('_lag')].copy()
    if not lag_feats_df.empty:
        lag_feats_df[['var', 'lag']] = lag_feats_df['Feature'].str.split('_lag', expand=True)
        lag_feats_df['lag'] = lag_feats_df['lag'].astype(int)
        lag_matrix = lag_feats_df.pivot(index='var', columns='lag', values='Importance').fillna(0)
        sns.heatmap(lag_matrix, ax=ax7, cmap='viridis', annot=False, fmt='.2f', cbar_kws={'label': '重要性'})
        ax7.set_title(f'{label} - 滞后特征重要性热力图', fontsize=13, fontweight='bold')
        ax7.set_xlabel('滞后天数')
        ax7.set_ylabel('变量')
    else:
        ax7.text(0.5, 0.5, '无滞后特征被选中', horizontalalignment='center', verticalalignment='center', fontsize=12)
        ax7.set_title(f'{label} - 滞后特征重要性热力图', fontsize=13, fontweight='bold')

    # 8. 滞后特征响应曲线
    ax8 = plt.subplot(3, 3, 8)
    for var in ['precipitation', 'temperature', 'swe']:
        var_lag_imp = core_feat_imp[core_feat_imp['Feature'].str.startswith(f"{var}_lag")]
        if not var_lag_imp.empty:
            var_lag_imp['lag'] = var_lag_imp['Feature'].str.split('_lag').str[1].astype(int)
            var_lag_imp = var_lag_imp.sort_values('lag')
            ax8.plot(var_lag_imp['lag'], var_lag_imp['Importance'], marker='o', label=var)
    ax8.set_xlabel('滞后天数')
    ax8.set_ylabel('特征重要性')
    ax8.set_title(f'{label} - 滞后响应曲线', fontsize=13, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{label}_Adaptive_Lag_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 步骤7: SHAP值解释
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_core)
    
    # 绘制SHAP摘要图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_core, feature_names=core_feats, plot_type="bar", show=False)
    plt.title(f'{label} - SHAP特征重要性', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{label}_SHAP_Summary_Bar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_core, feature_names=core_feats, show=False)
    plt.title(f'{label} - SHAP依赖图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{label}_SHAP_Summary_Dot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 步骤8: 保存结果
    results_df = pd.DataFrame({
        'y_true': y_full,
        'y_pred': y_full_pred,
        'residual': residuals
    })
    results_df.to_csv(f'results_{label}1.csv', index=False)
    
    # 保存选中的核心特征
    core_feat_imp.to_csv(f'selected_features_{label}_1..csv', index=False)
    
    return {
        'model': best_model,
        'core_features': core_feats,
        'core_feat_imp': core_feat_imp,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'residuals': residuals,
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals)
    }

# 7. 训练上下游自适应模型（无log变换）
# 上游模型
result_up = train_adaptive_gbm_no_log(
    X_train=X_train, y_train=y_train_up,
    X_test=X_test, y_test=y_test_up,
    X_full=df_full[X_cols_all].values, y_full=y_up,
    feature_names=X_cols_all, label="Upstream"
)

# 下游模型
result_down = train_adaptive_gbm_no_log(
    X_train=X_train, y_train=y_train_down,
    X_test=X_test, y_test=y_test_down,
    X_full=df_full[X_cols_all].values, y_full=y_down,
    feature_names=X_cols_all, label="Downstream"
)

# 8. 结果总结与输出
print("\n最终选择的核心滞后变量 (上游 Upstream):")
upstream_lag_vars = [var for var in result_up['core_features'] if 'lag' in var]
for var in upstream_lag_vars:
    print(var)

print("\n最终选择的核心滞后变量 (下游 Downstream):")
downstream_lag_vars = [var for var in result_down['core_features'] if 'lag' in var]
for var in downstream_lag_vars:
    print(var)
    
summary = pd.DataFrame({
    "指标": ["测试R²", "测试RMSE", "测试MAE", "测试MAPE(%)", "残差偏度", "残差峰度"],
    "Upstream": [
        f"{result_up['test_r2']:.4f}",
        f"{result_up['test_rmse']:.2f}",
        f"{result_up['test_mae']:.2f}",
        f"{result_up['test_mape']:.2f}",
        f"{result_up['skewness']:.4f}",
        f"{result_up['kurtosis']:.4f}"
    ],
    "Downstream": [
        f"{result_down['test_r2']:.4f}",
        f"{result_down['test_rmse']:.2f}",
        f"{result_down['test_mae']:.2f}",
        f"{result_down['test_mape']:.2f}",
        f"{result_down['skewness']:.4f}",
        f"{result_down['kurtosis']:.4f}"
    ]
})
print("\n" + summary.to_string(index=False))

# 保存残差用于copula分析
copula_residuals = pd.DataFrame({
    'date': df_full["date"],
    'upstream_residual': result_up['residuals'],
    'downstream_residual': result_down['residuals']
})
copula_residuals.to_csv('residuals_1.csv', index=False)
print("\n残差数据已保存到 'residuals_1.csv'")