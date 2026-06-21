import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, t, gaussian_kde, pearsonr, kendalltau
from scipy.special import gamma
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_t as multivariate_student_t
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 设置字体 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# ===================== 2. 加载数据 =====================
DATA_PATH = r"D:\降水与径流量\data\discharge_obs_pred_res_pseudo.csv"
df = pd.read_csv(DATA_PATH)

x = df["upstream_residual"].dropna().values
y = df["downstream_residual"].dropna().values
n = len(x)

# ===================== 3. 使用已有参数 =====================
alpha = 1.1322
beta = 0.3138
delta = 344.38
gamma_stable = 268.41

nu = 4.1199
mu_t = -2411.45
sigma_t = 22559.88

rho_cop = 0.3368
nu_cop = 3.3412

print("=" * 80)
print("二维残差联合密度对比：真实 vs 独立假设 vs Copula模型")
print("=" * 80)
print(f"样本数: {n}")
print(f"Copula参数: ρ={rho_cop:.4f}, ν={nu_cop:.4f}")
print()

# ===================== 4. 定义分布函数 =====================
def f_up(xx):
    return levy_stable.pdf(xx, alpha, beta, loc=delta, scale=gamma_stable)

def F_up(xx):
    return levy_stable.cdf(xx, alpha, beta, loc=delta, scale=gamma_stable)

def f_down(yy):
    return t.pdf(yy, df=nu, loc=mu_t, scale=sigma_t)

def F_down(yy):
    return t.cdf(yy, df=nu, loc=mu_t, scale=sigma_t)

def student_t_copula_pdf(u, v, rho, nu_c):
    u = np.clip(u, 1e-8, 1-1e-8)
    v = np.clip(v, 1e-8, 1-1e-8)
    x_t = t.ppf(u, nu_c)
    y_t = t.ppf(v, nu_c)
    term1 = gamma((nu_c+2)/2) * gamma(nu_c/2) / (gamma((nu_c+1)/2)**2)
    term2 = 1 / np.sqrt(1 - rho**2)
    term3 = (1 + (x_t**2 + y_t**2 - 2*rho*x_t*y_t) / (nu_c*(1-rho**2)))**(-((nu_c+2)/2))
    term4 = (1 + x_t**2/nu_c)**(-(nu_c+1)/2) * (1 + y_t**2/nu_c)**(-(nu_c+1)/2)
    return term1 * term2 * term3 / term4

def joint_density_independent(xx, yy):
    return f_up(xx) * f_down(yy)

def joint_density_copula(xx, yy):
    uu = F_up(xx)
    vv = F_down(yy)
    return f_up(xx) * f_down(yy) * student_t_copula_pdf(uu, vv, rho_cop, nu_cop)

kde = gaussian_kde([x, y])
def joint_density_true(xx, yy):
    pos = np.vstack([xx.ravel(), yy.ravel()])
    return kde(pos).reshape(xx.shape)

# ===================== 5. 定义经验分位数函数（用于模拟） =====================
x_sorted = np.sort(x)
y_sorted = np.sort(y)

def inv_x(u):
    return np.interp(u, np.linspace(0, 1, len(x_sorted)), x_sorted)

def inv_y(v):
    return np.interp(v, np.linspace(0, 1, len(y_sorted)), y_sorted)

# ===================== 6. 生成模拟样本（用于散点图） =====================
np.random.seed(42)
M = 5000
print(f"生成模拟样本 (M={M})...")

Sigma = np.array([[1, rho_cop], [rho_cop, 1]])

# Copula模型采样
x_cop = np.zeros((M, n))
y_cop = np.zeros((M, n))
for i in range(M):
    if (i + 1) % 1000 == 0:
        print(f"  Copula采样进度: {i+1}/{M}")
    rv = multivariate_student_t(loc=[0, 0], shape=Sigma, df=nu_cop)
    z = rv.rvs(size=n)
    u = t.cdf(z[:, 0], df=nu_cop)
    v = t.cdf(z[:, 1], df=nu_cop)
    x_cop[i, :] = inv_x(u)
    y_cop[i, :] = inv_y(v)

# 独立假设采样
x_ind = np.zeros((M, n))
y_ind = np.zeros((M, n))
for i in range(M):
    if (i + 1) % 1000 == 0:
        print(f"  独立采样进度: {i+1}/{M}")
    u = np.random.uniform(0, 1, n)
    v = np.random.uniform(0, 1, n)
    x_ind[i, :] = inv_x(u)
    y_ind[i, :] = inv_y(v)

# 展平用于散点图
x_cop_flat = x_cop.flatten()
y_cop_flat = y_cop.flatten()
x_ind_flat = x_ind.flatten()
y_ind_flat = y_ind.flatten()

print("采样完成！")
print()

# ===================== 7. 评估指标 =====================
print("=" * 60)
print("评估指标")
print("=" * 60)

# 7.1 相关系数
rho_pearson, _ = pearsonr(x, y)
tau_kendall, _ = kendalltau(x, y)

rho_cop_pearson, _ = pearsonr(x_cop_flat, y_cop_flat)
tau_cop_kendall, _ = kendalltau(x_cop_flat, y_cop_flat)

rho_ind_pearson, _ = pearsonr(x_ind_flat, y_ind_flat)
tau_ind_kendall, _ = kendalltau(x_ind_flat, y_ind_flat)

print(f"\nPearson相关系数:")
print(f"  原始残差:     {rho_pearson:.4f}")
print(f"  Copula模型:   {rho_cop_pearson:.4f}")
print(f"  独立假设:     {rho_ind_pearson:.4f}")

print(f"\nKendall秩相关系数:")
print(f"  原始残差:     {tau_kendall:.4f}")
print(f"  Copula模型:   {tau_cop_kendall:.4f}")
print(f"  独立假设:     {tau_ind_kendall:.4f}")

# 7.2 尾部依赖系数
def tail_dependence(x, y, p=0.95, tail='upper'):
    if tail == 'upper':
        x_thresh = np.percentile(x, p * 100)
        y_thresh = np.percentile(y, p * 100)
        idx = x > x_thresh
    else:
        x_thresh = np.percentile(x, (1-p) * 100)
        y_thresh = np.percentile(y, (1-p) * 100)
        idx = x < x_thresh
    if np.sum(idx) > 0:
        return np.mean(y[idx] > y_thresh) if tail == 'upper' else np.mean(y[idx] < y_thresh)
    return 0.0

lambda_u_obs = tail_dependence(x, y, tail='upper')
lambda_l_obs = tail_dependence(x, y, tail='lower')
lambda_u_cop = tail_dependence(x_cop_flat, y_cop_flat, tail='upper')
lambda_l_cop = tail_dependence(x_cop_flat, y_cop_flat, tail='lower')
lambda_u_ind = tail_dependence(x_ind_flat, y_ind_flat, tail='upper')
lambda_l_ind = tail_dependence(x_ind_flat, y_ind_flat, tail='lower')

print(f"\n上尾依赖系数 λ_U (95%分位数):")
print(f"  原始残差:     {lambda_u_obs:.4f}")
print(f"  Copula模型:   {lambda_u_cop:.4f}")
print(f"  独立假设:     {lambda_u_ind:.4f}")

print(f"\n下尾依赖系数 λ_L (5%分位数):")
print(f"  原始残差:     {lambda_l_obs:.4f}")
print(f"  Copula模型:   {lambda_l_cop:.4f}")
print(f"  独立假设:     {lambda_l_ind:.4f}")

# 7.3 极端事件联合概率
thresh = 0.95
x_thresh = np.percentile(x, thresh * 100)
y_thresh = np.percentile(y, thresh * 100)

prob_obs = np.mean((x > x_thresh) & (y > y_thresh))
prob_cop = np.mean((x_cop_flat > x_thresh) & (y_cop_flat > y_thresh))
prob_ind = np.mean((x_ind_flat > x_thresh) & (y_ind_flat > y_thresh))

print(f"\n极端事件联合概率 P(ε1 > 95%分位数, ε2 > 95%分位数):")
print(f"  原始残差:     {prob_obs:.6f}")
print(f"  Copula模型:   {prob_cop:.6f}")
print(f"  独立假设:     {prob_ind:.6f}")

# 7.4 对数似然、AIC、BIC
loglik_true = np.sum(np.log(joint_density_true(x, y) + 1e-10))
loglik_ind = np.sum(np.log(joint_density_independent(x, y) + 1e-10))
loglik_cop = np.sum(np.log(joint_density_copula(x, y) + 1e-10))

k_ind = 4 + 3
k_cop = k_ind + 2

aic_ind = 2 * k_ind - 2 * loglik_ind
aic_cop = 2 * k_cop - 2 * loglik_cop
bic_ind = k_ind * np.log(n) - 2 * loglik_ind
bic_cop = k_cop * np.log(n) - 2 * loglik_cop

print(f"\n对数似然、AIC、BIC:")
print(f"{'模型':<20} {'对数似然':<15} {'AIC':<12} {'BIC':<12}")
print("-" * 60)
print(f"{'真实数据 (KDE)':<20} {loglik_true:<15.2f} {'—':<12} {'—':<12}")
print(f"{'独立假设':<20} {loglik_ind:<15.2f} {aic_ind:<12.2f} {bic_ind:<12.2f}")
print(f"{'Copula模型':<20} {loglik_cop:<15.2f} {aic_cop:<12.2f} {bic_cop:<12.2f}")

print(f"\nCopula vs 独立: ΔAIC = {aic_cop - aic_ind:+.2f}, ΔBIC = {bic_cop - bic_ind:+.2f}")
print()

# ===================== 8. 生成网格（用于密度图） =====================
x_min = np.percentile(x, 2)
x_max = np.percentile(x, 98)
y_min = np.percentile(y, 2)
y_max = np.percentile(y, 98)

x_grid = np.linspace(x_min, x_max, 80)
y_grid = np.linspace(y_min, y_max, 80)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

Z_true = joint_density_true(X_grid, Y_grid)
Z_ind = joint_density_independent(X_grid, Y_grid)
Z_cop = joint_density_copula(X_grid, Y_grid)

vmin = 0
vmax = max(Z_true.max(), Z_ind.max(), Z_cop.max())
levels = np.linspace(vmin, vmax, 12)

# ===================== 9. 图1：等高线图（三组叠加） =====================
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))

c1 = ax1.contour(X_grid, Y_grid, Z_true, levels=levels, colors='red', linewidths=2, alpha=0.8, linestyles='-')
c2 = ax1.contour(X_grid, Y_grid, Z_ind, levels=levels, colors='green', linewidths=2, alpha=0.8, linestyles='--')
c3 = ax1.contour(X_grid, Y_grid, Z_cop, levels=levels, colors='blue', linewidths=2, alpha=0.8, linestyles='-.')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, linestyle='-', label='真实数据 (KDE)'),
    Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='独立假设'),
    Line2D([0], [0], color='blue', linewidth=2, linestyle='-.', label=f'Copula模型 (ρ={rho_cop:.3f})')
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)

ax1.set_xlabel('上游残差', fontsize=12)
ax1.set_ylabel('下游残差', fontsize=12)
ax1.set_title('二维联合密度等高线对比', fontsize=14, fontweight='bold')
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('contour_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== 10. 图2：热力图（三组并排） =====================
fig2, axes = plt.subplots(1, 3, figsize=(18, 5.5))

im1 = axes[0].imshow(Z_true.T, origin='lower', 
                      extent=[x_min, x_max, y_min, y_max], 
                      aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title('真实数据 (KDE)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('上游残差', fontsize=10)
axes[0].set_ylabel('下游残差', fontsize=10)
plt.colorbar(im1, ax=axes[0], shrink=0.8, label='密度')

im2 = axes[1].imshow(Z_ind.T, origin='lower', 
                      extent=[x_min, x_max, y_min, y_max], 
                      aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title('独立假设', fontsize=12, fontweight='bold')
axes[1].set_xlabel('上游残差', fontsize=10)
axes[1].set_ylabel('下游残差', fontsize=10)
plt.colorbar(im2, ax=axes[1], shrink=0.8, label='密度')

im3 = axes[2].imshow(Z_cop.T, origin='lower', 
                      extent=[x_min, x_max, y_min, y_max], 
                      aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
axes[2].set_title(f'Copula模型 (ρ={rho_cop:.3f})', fontsize=12, fontweight='bold')
axes[2].set_xlabel('上游残差', fontsize=10)
axes[2].set_ylabel('下游残差', fontsize=10)
plt.colorbar(im3, ax=axes[2], shrink=0.8, label='密度')

plt.suptitle('二维联合密度热力图对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== 11. 图3：3D曲面图 =====================
fig3 = plt.figure(figsize=(18, 6))

ax1 = fig3.add_subplot(1, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X_grid, Y_grid, Z_true, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
ax1.set_title('(a) 真实数据', fontsize=12, fontweight='bold')
ax1.set_xlabel('上游残差', fontsize=9)
ax1.set_ylabel('下游残差', fontsize=9)
ax1.set_zlabel('密度', fontsize=9)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_zlim(vmin, vmax)
ax1.view_init(elev=30, azim=30)

ax2 = fig3.add_subplot(1, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X_grid, Y_grid, Z_ind, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
ax2.set_title('(b) 独立假设', fontsize=12, fontweight='bold')
ax2.set_xlabel('上游残差', fontsize=9)
ax2.set_ylabel('下游残差', fontsize=9)
ax2.set_zlabel('密度', fontsize=9)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_zlim(vmin, vmax)
ax2.view_init(elev=30, azim=30)

ax3 = fig3.add_subplot(1, 3, 3, projection='3d')
surf3 = ax3.plot_surface(X_grid, Y_grid, Z_cop, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
ax3.set_title(f'(c) Copula模型 (ρ={rho_cop:.3f})', fontsize=12, fontweight='bold')
ax3.set_xlabel('上游残差', fontsize=9)
ax3.set_ylabel('下游残差', fontsize=9)
ax3.set_zlabel('密度', fontsize=9)
ax3.set_xlim(x_min, x_max)
ax3.set_ylim(y_min, y_max)
ax3.set_zlim(vmin, vmax)
ax3.view_init(elev=30, azim=30)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.suptitle('二维联合密度3D曲面图对比', fontsize=14, fontweight='bold', y=0.98)
plt.savefig('3d_surface_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== 12. 图4：残差散点图对比（三组并排） =====================
# 采样部分点避免过密
sample_size = min(5000, len(x_cop_flat))
sample_idx = np.random.choice(len(x_cop_flat), sample_size, replace=False)

fig4, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 原始残差
ax1 = axes[0]
ax1.scatter(x, y, s=5, alpha=0.3, color='blue', edgecolors='none')
ax1.set_xlabel('上游残差 ε1', fontsize=12)
ax1.set_ylabel('下游残差 ε2', fontsize=12)
ax1.set_title(f'原始残差\nPearson ρ={rho_pearson:.3f}, Kendall τ={tau_kendall:.3f}', 
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 独立假设
ax2 = axes[1]
ax2.scatter(x_ind_flat[sample_idx], y_ind_flat[sample_idx], s=5, alpha=0.3, color='green', edgecolors='none')
ax2.set_xlabel('上游残差 ε1', fontsize=12)
ax2.set_ylabel('下游残差 ε2', fontsize=12)
ax2.set_title(f'独立假设\nPearson ρ={rho_ind_pearson:.3f}, Kendall τ={tau_ind_kendall:.3f}', 
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Copula模型
ax3 = axes[2]
ax3.scatter(x_cop_flat[sample_idx], y_cop_flat[sample_idx], s=5, alpha=0.3, color='red', edgecolors='none')
ax3.set_xlabel('上游残差 ε1', fontsize=12)
ax3.set_ylabel('下游残差 ε2', fontsize=12)
ax3.set_title(f'Copula模型\nPearson ρ={rho_cop_pearson:.3f}, Kendall τ={tau_cop_kendall:.3f}', 
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.suptitle('残差联合分布对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('residual_scatter_comparison.png', dpi=300, bbox_inches='tight')
plt.show()





