import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, t, multivariate_t, kendalltau, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 基础配置 =====================
# 字体设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 数据路径
RES_PATH = r"D:\降水与径流量\data\discharge_obs_pred_res_pseudo.csv"

# Student-t Copula参数
rho = 0.3368
nu_cop = 3.3412
n_sim = 10000

# 上游α-stable参数
alpha_up = 1.1322
beta_up = 0.3138
delta_up = 344.38
gamma_up = 268.41

# 下游t分布参数
nu_down = 4.1199
mu_down = -2411.45
sigma_down = 22559.88

# ===================== 2. 加载实测数据 =====================
df = pd.read_csv(RES_PATH)
# 提取实测残差（过滤空值）
res1 = df["upstream_residual"].dropna().values  # 上游实测残差
res2 = df["downstream_residual"].dropna().values  # 下游实测残差

# 直接使用文件内的伪观测值u/v
u = df["u"].values  
v = df["v"].values

# ===================== 3. Copula模拟残差 =====================
# 3.1 构建协方差矩阵
Sigma = np.array([[1, rho],
                  [rho, 1]])

# 3.2 生成t-copula样本
np.random.seed(42)  
rv = multivariate_t(loc=[0,0], shape=Sigma, df=nu_cop) 
z = rv.rvs(size=n_sim)

# 3.3 转换为均匀分布
u_sim = t.cdf(z[:,0], df=nu_cop)
v_sim = t.cdf(z[:,1], df=nu_cop)

# ===================== 4. 逆变换得到模拟残差 =====================
# 上游：α-stable逆变换
eps1_sim = levy_stable.ppf(u_sim, alpha_up, beta_up, loc=delta_up, scale=gamma_up)
# 下游：t分布逆变换
eps2_sim = t.ppf(v_sim, df=nu_down, loc=mu_down, scale=sigma_down)

# ===================== 5. 核心验证1：Kendall τ（依赖结构验证） =====================
tau_obs, p_tau_obs = kendalltau(res1, res2)
tau_sim, p_tau_sim = kendalltau(eps1_sim, eps2_sim)

# ===================== 6. 核心验证2：Spearman ρ（秩相关验证） =====================
rho_obs, p_rho_obs = spearmanr(res1, res2)
rho_sim, p_rho_sim = spearmanr(eps1_sim, eps2_sim)

# ===================== 7. 打印所有验证结果（量化指标） =====================
print("="*80)
print("1. Kendall τ (依赖结构验证)")
print(f"   实测残差 Kendall τ: {tau_obs:.4f} (p值: {p_tau_obs:.6f})")
print(f"   模拟残差 Kendall τ: {tau_sim:.4f} (p值: {p_tau_sim:.6f})")
print(f"   相对误差: {abs(tau_sim - tau_obs)/tau_obs*100:.2f}%")

print("\n2. Spearman ρ (秩相关验证)")
print(f"   实测残差 Spearman ρ: {rho_obs:.4f} (p值: {p_rho_obs:.6f})")
print(f"   模拟残差 Spearman ρ: {rho_sim:.4f} (p值: {p_rho_sim:.6f})")
print(f"   相对误差: {abs(rho_sim - rho_obs)/rho_obs*100:.2f}%")

# 极端概率验证
thr1 = np.quantile(res1, 0.95)
thr2 = np.quantile(res2, 0.95)
empirical_prob = np.mean((res1 > thr1) & (res2 > thr2))
copula_prob = np.mean((eps1_sim > thr1) & (eps2_sim > thr2))

print("\n3. 95%极端联合概率验证")
print(f"   上游残差95%阈值: {thr1:.4f}")
print(f"   下游残差95%阈值: {thr2:.4f}")
print(f"   实测极端概率: {empirical_prob:.6f}")
print(f"   模拟极端概率: {copula_prob:.6f}")
print(f"   相对误差: {abs(copula_prob - empirical_prob)/empirical_prob*100:.2f}%")
print("="*80)

# ===================== 8. 可视化验证（统一轴刻度） =====================
# 计算全局轴范围（保证所有对比图刻度一致）
x_min = min(res1.min(), eps1_sim.min())
x_max = max(res1.max(), eps1_sim.max())
y_min = min(res2.min(), eps2_sim.min())
y_max = max(res2.max(), eps2_sim.max())

# -------------------- 图1：残差散点对比（统一轴刻度） --------------------
fig1 = plt.figure(figsize=(12, 5), constrained_layout=True)

# 8.1 子图1：实测残差散点
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(res1, res2, s=3, alpha=0.4, color="#1f77b4")
ax1.set_title("Observed Residuals", fontsize=12, fontweight="bold")
ax1.set_xlabel("Upstream Residual", fontsize=10)
ax1.set_ylabel("Downstream Residual", fontsize=10)
ax1.grid(alpha=0.3)
# 统一轴范围
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)

# 8.2 子图2：模拟残差散点
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(eps1_sim, eps2_sim, s=3, alpha=0.4, color="#ff7f0e")
ax2.set_title("Simulated Residuals (Student-t Copula)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Upstream Residual", fontsize=10)
ax2.set_ylabel("Downstream Residual", fontsize=10)
ax2.grid(alpha=0.3)
# 统一轴范围（和左图完全一致）
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)

# 图1标题
fig1.suptitle("Comparison of Residual Scatter Distribution", fontsize=14, fontweight="bold")
# 保存代码已注释
# plt.savefig("fig1_residual_scatter.png", bbox_inches="tight", facecolor="white", dpi=300)
plt.show()

# -------------------- 图2：联合密度对比（统一轴刻度） --------------------
fig2 = plt.figure(figsize=(12, 6), constrained_layout=True)

# 8.3 子图3：实测残差联合密度
ax3 = plt.subplot(1, 2, 1)
hex1 = ax3.hexbin(res1, res2, gridsize=40, cmap="Blues", mincnt=1)
ax3.set_title("Observed Residuals", fontsize=12, fontweight="bold")
ax3.set_xlabel("Upstream Residual", fontsize=10)
ax3.set_ylabel("Downstream Residual", fontsize=10)
plt.colorbar(hex1, ax=ax3, shrink=0.8, label="Count")
# 统一轴范围
ax3.set_xlim(x_min, x_max)
ax3.set_ylim(y_min, y_max)

# 8.4 子图4：模拟残差联合密度
ax4 = plt.subplot(1, 2, 2)
hex2 = ax4.hexbin(eps1_sim, eps2_sim, gridsize=40, cmap="Oranges", mincnt=1)
ax4.set_title("Simulated Residuals (Student-t Copula)", fontsize=12, fontweight="bold")
ax4.set_xlabel("Upstream Residual", fontsize=10)
ax4.set_ylabel("Downstream Residual", fontsize=10)
plt.colorbar(hex2, ax=ax4, shrink=0.8, label="Count")
# 统一轴范围
ax4.set_xlim(x_min, x_max)
ax4.set_ylim(y_min, y_max)

# 图2标题
fig2.suptitle("Comparison of Residual Joint Density Distribution", fontsize=14, fontweight="bold")
# 保存代码已注释
# plt.savefig("fig2_residual_joint_density.png", bbox_inches="tight", facecolor="white", dpi=300)
plt.show()

# -------------------- 图3：极端区域对比（统一轴刻度） --------------------
fig3 = plt.figure(figsize=(12, 5), constrained_layout=True)

# 8.5 子图5：极端区域对比（实测）
ax5 = plt.subplot(1, 2, 1)
ax5.scatter(res1, res2, s=3, alpha=0.4, color="#1f77b4")
ax5.axvline(thr1, color="red", linestyle="--", linewidth=1, label="95% Quantile")
ax5.axhline(thr2, color="red", linestyle="--", linewidth=1)
ax5.fill_betweenx([thr2, res2.max()], thr1, res1.max(), color="red", alpha=0.2, label="Extreme Region")
ax5.set_title(f"Observed (Extreme Prob={empirical_prob:.6f})", fontsize=11)
ax5.set_xlabel("Upstream Residual", fontsize=10)
ax5.set_ylabel("Downstream Residual", fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)
# 统一轴范围
ax5.set_xlim(x_min, x_max)
ax5.set_ylim(y_min, y_max)

# 8.6 子图6：极端区域对比（模拟）
ax6 = plt.subplot(1, 2, 2)
ax6.scatter(eps1_sim, eps2_sim, s=3, alpha=0.4, color="#ff7f0e")
ax6.axvline(thr1, color="red", linestyle="--", linewidth=1, label="95% Quantile")
ax6.axhline(thr2, color="red", linestyle="--", linewidth=1)
ax6.fill_betweenx([thr2, eps2_sim.max()], thr1, eps1_sim.max(), color="red", alpha=0.2, label="Extreme Region")
ax6.set_title(f"Simulated (Extreme Prob={copula_prob:.6f})", fontsize=11)
ax6.set_xlabel("Upstream Residual", fontsize=10)
ax6.set_ylabel("Downstream Residual", fontsize=10)
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)
# 统一轴范围
ax6.set_xlim(x_min, x_max)
ax6.set_ylim(y_min, y_max)

# 图3标题
fig3.suptitle("Comparison of Extreme Region Distribution (95% Quantile)", fontsize=14, fontweight="bold")
# 保存代码已注释
# plt.savefig("fig3_residual_extreme_region.png", bbox_inches="tight", facecolor="white", dpi=300)
plt.show()

# ===================== 9. 补充：Kendall τ/Spearman ρ可视化（无需统一轴） =====================
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# 9.1 Kendall τ对比
ax1.bar(["Observed", "Simulated"], [tau_obs, tau_sim], color=["#1f77b4", "#ff7f0e"], alpha=0.7)
ax1.set_title(f"Kendall τ Comparison (Error={abs(tau_sim-tau_obs)/tau_obs*100:.2f}%)", fontsize=12)
ax1.set_ylabel("Kendall τ", fontsize=10)
ax1.grid(alpha=0.3, axis="y")
# 标注数值
for i, v in enumerate([tau_obs, tau_sim]):
    ax1.text(i, v+0.005, f"{v:.4f}", ha="center", fontsize=9)

# 9.2 Spearman ρ对比
ax2.bar(["Observed", "Simulated"], [rho_obs, rho_sim], color=["#1f77b4", "#ff7f0e"], alpha=0.7)
ax2.set_title(f"Spearman ρ Comparison (Error={abs(rho_sim-rho_obs)/rho_obs*100:.2f}%)", fontsize=12)
ax2.set_ylabel("Spearman ρ", fontsize=10)
ax2.grid(alpha=0.3, axis="y")
# 标注数值
for i, v in enumerate([rho_obs, rho_sim]):
    ax2.text(i, v+0.005, f"{v:.4f}", ha="center", fontsize=9)

# 图4标题
fig4.suptitle("Comparison of Rank Correlation Coefficients", fontsize=14, fontweight="bold")
# 保存代码已注释
# plt.savefig("fig4_correlation_coefficients.png", bbox_inches="tight", facecolor="white", dpi=300)
plt.show()