import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import levy_stable, t
import warnings
warnings.filterwarnings('ignore')

# 1. 加载残差数据
# 假设您已保存残差数据
residuals_df = pd.read_csv(r"D:\降水与径流量\data\residuals_1.csv")  # 或您的残差文件路径
up_residual = residuals_df['upstream_residual'].values
down_residual = residuals_df['downstream_residual'].values

# 2. 定义上游α-stable分布
alpha_up = 1.132207
beta_up = 0.313796
gamma_up = 268.412047  
delta_up = 344.383956  

up_dist = levy_stable(alpha=alpha_up, beta=beta_up, loc=delta_up, scale=gamma_up)

# 3. 定义下游Student-t分布
df_down = 4.119946
loc_down = -2411.450470
scale_down = 22559.882666

down_dist = t(df=df_down, loc=loc_down, scale=scale_down)

# 4. 概率积分变换：U = F(ε)

# 计算累积分布函数值
u = up_dist.cdf(up_residual)
v = down_dist.cdf(down_residual)

# 5. 变换结果检验
print("\n变换结果检验:")

# 5.1 基本统计量检验（应接近[0,1]均匀分布）
print("\n u 统计量:")
print(f"  均值: {u.mean():.4f} ")
print(f"  标准差: {u.std():.4f} ")
print(f"  最小值: {u.min():.4f}")
print(f"  最大值: {u.max():.4f}")

print("\n v 统计量:")
print(f"  均值: {v.mean():.4f} ")
print(f"  标准差: {v.std():.4f} ")
print(f"  最小值: {v.min():.4f}")
print(f"  最大值: {v.max():.4f}")

# 5.2 均匀性检验（Kolmogorov-Smirnov检验）
ks_up = stats.kstest(u, 'uniform')
ks_down = stats.kstest(v, 'uniform')

print("\n均匀性KS检验:")
print(f"  上游u: KS统计量={ks_up.statistic:.4f}, p值={ks_up.pvalue:.4f}")
print(f"  下游v: KS统计量={ks_down.statistic:.4f}, p值={ks_down.pvalue:.4f}")

if ks_up.pvalue > 0.05 and ks_down.pvalue > 0.05:
    print("  ✓ 变换成功：序列服从均匀分布 (p>0.05)")
else:
    print("  ⚠️ 注意：序列与均匀分布存在显著差异")

# 5.3 可视化检验
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 上游U的直方图
axes[0, 0].hist(u, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axhline(y=1, color='red', linestyle='--', label='理论均匀分布')
axes[0, 0].set_xlabel('u')
axes[0, 0].set_ylabel('密度')
axes[0, 0].set_title('上游直方图')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 下游U的直方图
axes[0, 1].hist(v, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
axes[0, 1].axhline(y=1, color='red', linestyle='--', label='理论均匀分布')
axes[0, 1].set_xlabel('v')
axes[0, 1].set_ylabel('密度')
axes[0, 1].set_title('下游直方图')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# U_up vs U_down 散点图
axes[0, 2].scatter(u, v, alpha=0.3, s=2, c='purple')
axes[0, 2].set_xlabel('U_up')
axes[0, 2].set_ylabel('U_down')
axes[0, 2].set_title('U序列散点图')
axes[0, 2].grid(True, alpha=0.3)

# Q-Q图：U_up vs 均匀分布
axes[1, 0].plot(np.sort(u), np.linspace(0, 1, len(u)), 'o', markersize=2, alpha=0.5)
axes[1, 0].plot([0, 1], [0, 1], 'r--', linewidth=2)
axes[1, 0].set_xlabel('理论分位数 (均匀分布)')
axes[1, 0].set_ylabel('样本分位数')
axes[1, 0].set_title('上游Q-Q图')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q图：U_down vs 均匀分布
axes[1, 1].plot(np.sort(v), np.linspace(0, 1, len(v)), 'o', markersize=2, alpha=0.5)
axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2)
axes[1, 1].set_xlabel('理论分位数 (均匀分布)')
axes[1, 1].set_ylabel('样本分位数')
axes[1, 1].set_title('下游Q-Q图')
axes[1, 1].grid(True, alpha=0.3)

# 自相关函数（检验独立性）
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(u, lags=40, ax=axes[1, 2], alpha=0.05)
axes[1, 2].set_title('上游自相关函数')

plt.tight_layout()
plt.savefig('probability_integral_transform_check.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 6. 保存变换结果用于Copula分析
# ============================================================
U_df = pd.DataFrame({
    'date': residuals_df['date'] if 'date' in residuals_df.columns else range(len(u)),
    'u': u,
    'v': v,
    'epsilon_up': up_residual,
    'epsilon_down': down_residual
})
U_df.to_csv('pseudo_obs.csv', index=False)
