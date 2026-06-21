import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import stats
from scipy.special import gamma
from scipy.optimize import minimize
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']  
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data_path = r'D:\降水与径流量\data\pit_series_full_harmonic_powell.csv'
df = pd.read_csv(data_path)
u = df['upstream_pit'].values
v = df['downstream_pit'].values
n = len(u)

print(f"数据点数: {n}")
print(f"U的范围: [{u.min():.4f}, {u.max():.4f}]")
print(f"V的范围: [{v.min():.4f}, {v.max():.4f}]")
print(f"Kendall's tau: {stats.kendalltau(u, v)[0]:.4f}")
print(f"Spearman's rho: {stats.spearmanr(u, v)[0]:.4f}")

# Copula函数定义

# 1. Gaussian Copula
def gaussian_copula_pdf(u, v, rho):
    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)
    det = 1 - rho**2
    exponent = -(x**2 - 2*rho*x*y + y**2) / (2*det) + (x**2 + y**2)/2
    return np.exp(exponent) / np.sqrt(det)

def gaussian_copula_loglik(rho, u, v):
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    pdf_vals = gaussian_copula_pdf(u, v, rho)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    return -np.sum(np.log(pdf_vals))

# 2. Student-t Copula
def student_t_copula_pdf(u, v, rho, nu):
    x = stats.t.ppf(u, nu)
    y = stats.t.ppf(v, nu)
    term1 = gamma((nu+2)/2) * gamma(nu/2) / (gamma((nu+1)/2)**2)
    term2 = 1 / np.sqrt(1 - rho**2)
    term3 = (1 + (x**2 + y**2 - 2*rho*x*y) / (nu*(1-rho**2)))**(-((nu+2)/2))
    term4 = (1 + x**2/nu)**(-(nu+1)/2) * (1 + y**2/nu)**(-(nu+1)/2)
    return term1 * term2 * term3 / term4

def student_t_copula_loglik(params, u, v):
    rho, nu = params
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    pdf_vals = student_t_copula_pdf(u, v, rho, nu)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    return -np.sum(np.log(pdf_vals))

# 3. Clayton Copula
def clayton_copula_pdf(u, v, theta):
    if theta <= 0:
        return np.zeros_like(u)
    term1 = (1 + theta) * (u*v)**(-1-theta)
    term2 = (u**(-theta) + v**(-theta) - 1)**(-(2+1/theta))
    return term1 * term2

def clayton_copula_loglik(theta, u, v):
    if theta <= 1e-6:
        return 1e10
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    pdf_vals = clayton_copula_pdf(u, v, theta)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    return -np.sum(np.log(pdf_vals))

# 4. Gumbel Copula
def gumbel_copula_pdf(u, v, theta):
    if theta < 1:
        return np.zeros_like(u)
    A = (-np.log(u))**theta + (-np.log(v))**theta
    C = np.exp(-A**(1/theta))
    term1 = C / (u * v)
    term2 = A**(-2 + 2/theta)
    term3 = (np.log(u) * np.log(v))**(theta-1)
    term4 = 1 + (theta - 1) * A**(-1/theta)
    return term1 * term2 * term3 * term4

def gumbel_copula_loglik(theta, u, v):
    if theta < 1:
        return 1e10
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    pdf_vals = gumbel_copula_pdf(u, v, theta)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    return -np.sum(np.log(pdf_vals))

# 5. Joe Copula
def joe_copula_pdf(u, v, theta):
    if theta < 1:
        return np.zeros_like(u)
    a = 1 - u
    b = 1 - v
    A = a ** (theta - 1)
    B = b ** (theta - 1)
    D = a**theta + b**theta - a**theta * b**theta
    E = D**(1/theta - 2)
    F = D + theta - 1
    
    return A * B * E * F


def joe_copula_loglik(theta, u, v):
    if theta < 1:
        return 1e10
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    pdf_vals = joe_copula_pdf(u, v, theta)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    return -np.sum(np.log(pdf_vals))

# 尾部依赖系数计算

def upper_tail_dependence(copula_type, params):
    """计算上尾依赖系数 λ_U"""
    if copula_type == 'Gaussian':
        return 0.0
    elif copula_type == 'Student-t':
        rho = params['rho']
        nu = params['nu']
        from scipy.stats import t as t_dist
        lambda_u = 2 * t_dist.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
        return lambda_u
    elif copula_type == 'Clayton':
        return 0.0
    elif copula_type == 'Gumbel':
        theta = params['theta']
        lambda_u = 2 - 2**(1/theta)
        return lambda_u
    elif copula_type == 'Joe':
        theta = params['theta']
        lambda_u = 2 - 2**(1/theta)
        return lambda_u
    return None

def lower_tail_dependence(copula_type, params):
    """计算下尾依赖系数 λ_L"""
    if copula_type == 'Gaussian':
        return 0.0
    elif copula_type == 'Student-t':
        rho = params['rho']
        nu = params['nu']
        from scipy.stats import t as t_dist
        lambda_l = 2 * t_dist.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
        return lambda_l
    elif copula_type == 'Clayton':
        theta = params['theta']
        if theta > 0:
            lambda_l = 2**(-1/theta)
        else:
            lambda_l = 0.0
        return lambda_l
    elif copula_type == 'Gumbel':
        return 0.0
    elif copula_type == 'Joe':
        return 0.0
    return None

# 模型拟合

results = {}

# 1. Gaussian Copula
print("\n1. Gaussian Copulan拟合：")
res_gauss = minimize(gaussian_copula_loglik, x0=[0.5], args=(u, v), 
                     bounds=[(-0.99, 0.99)], method='L-BFGS-B')
rho_gauss = res_gauss.x[0]
loglik_gauss = -res_gauss.fun
aic_gauss = 2 * 1 - 2 * loglik_gauss
bic_gauss = np.log(n) * 1 - 2 * loglik_gauss
results['Gaussian'] = {
    'params': {'rho': rho_gauss},
    'loglik': loglik_gauss,
    'AIC': aic_gauss,
    'BIC': bic_gauss,
    'n_params': 1
}
print(f"   参数: rho = {rho_gauss:.4f}")
print(f"   Log-likelihood: {loglik_gauss:.2f}")
print(f"   AIC: {aic_gauss:.2f}, BIC: {bic_gauss:.2f}")

# 2. Student-t Copula
print("\n2. 拟合 Student-t Copula...")
res_t = minimize(student_t_copula_loglik, x0=[0.5, 5], args=(u, v),
                 bounds=[(-0.99, 0.99), (2.1, 30)], method='L-BFGS-B')
rho_t, nu_t = res_t.x
loglik_t = -res_t.fun
aic_t = 2 * 2 - 2 * loglik_t
bic_t = np.log(n) * 2 - 2 * loglik_t
results['Student-t'] = {
    'params': {'rho': rho_t, 'nu': nu_t},
    'loglik': loglik_t,
    'AIC': aic_t,
    'BIC': bic_t,
    'n_params': 2
}
print(f"   参数: rho = {rho_t:.4f}, nu = {nu_t:.4f}")
print(f"   Log-likelihood: {loglik_t:.2f}")
print(f"   AIC: {aic_t:.2f}, BIC: {bic_t:.2f}")

# 3. Clayton Copula
print("\n3. Clayton Copula拟合")
res_clayton = minimize(clayton_copula_loglik, x0=[1.0], args=(u, v),
                       bounds=[(0.01, 20)], method='L-BFGS-B')
theta_clayton = res_clayton.x[0]
loglik_clayton = -res_clayton.fun
aic_clayton = 2 * 1 - 2 * loglik_clayton
bic_clayton = np.log(n) * 1 - 2 * loglik_clayton
results['Clayton'] = {
    'params': {'theta': theta_clayton},
    'loglik': loglik_clayton,
    'AIC': aic_clayton,
    'BIC': bic_clayton,
    'n_params': 1
}
print(f"   参数: theta = {theta_clayton:.4f}")
print(f"   Log-likelihood: {loglik_clayton:.2f}")
print(f"   AIC: {aic_clayton:.2f}, BIC: {bic_clayton:.2f}")

# 4. Gumbel Copula
print("\n4. Gumbel Copula拟合：")
res_gumbel = minimize(gumbel_copula_loglik, x0=[1.5], args=(u, v),
                      bounds=[(1.01, 20)], method='L-BFGS-B')
theta_gumbel = res_gumbel.x[0]
loglik_gumbel = -res_gumbel.fun
aic_gumbel = 2 * 1 - 2 * loglik_gumbel
bic_gumbel = np.log(n) * 1 - 2 * loglik_gumbel
results['Gumbel'] = {
    'params': {'theta': theta_gumbel},
    'loglik': loglik_gumbel,
    'AIC': aic_gumbel,
    'BIC': bic_gumbel,
    'n_params': 1
}
print(f"   参数: theta = {theta_gumbel:.4f}")
print(f"   Log-likelihood: {loglik_gumbel:.2f}")
print(f"   AIC: {aic_gumbel:.2f}, BIC: {bic_gumbel:.2f}")

# 5. Joe Copula
print("\n5. Joe Copula拟合：")
res_joe = minimize(joe_copula_loglik, x0=[2.0], args=(u, v),
                   bounds=[(1.01, 20)], method='L-BFGS-B')
theta_joe = res_joe.x[0]
loglik_joe = -res_joe.fun
aic_joe = 2 * 1 - 2 * loglik_joe
bic_joe = np.log(n) * 1 - 2 * loglik_joe
results['Joe'] = {
    'params': {'theta': theta_joe},
    'loglik': loglik_joe,
    'AIC': aic_joe,
    'BIC': bic_joe,
    'n_params': 1
}
print(f"   参数: theta = {theta_joe:.4f}")
print(f"   Log-likelihood: {loglik_joe:.2f}")
print(f"   AIC: {aic_joe:.2f}, BIC: {bic_joe:.2f}")

# 计算尾部依赖系数
print("\n尾部依赖系数分析：")
print(f"{'Copula':<12} {'上尾依赖 λ_U':<15} {'下尾依赖 λ_L':<15} {'解释'}")

tail_deps = {}
for model in results.keys():
    lambda_u = upper_tail_dependence(model, results[model]['params'])
    lambda_l = lower_tail_dependence(model, results[model]['params'])
    tail_deps[model] = {'upper': lambda_u, 'lower': lambda_l}
    results[model]['upper_tail'] = lambda_u
    results[model]['lower_tail'] = lambda_l
    
    # 格式化输出
    lu_str = f"{lambda_u:.6f}" if lambda_u is not None else "N/A"
    ll_str = f"{lambda_l:.6f}" if lambda_l is not None else "N/A"
    
    # 添加解释
    explanation = ""
    if lambda_u > 0.05 and lambda_l > 0.05:
        explanation = "双尾依赖"
    elif lambda_u > 0.05:
        explanation = "上尾依赖"
    elif lambda_l > 0.05:
        explanation = "下尾依赖"
    else:
        explanation = "无尾依赖"
    
    print(f"    {model:<12} {lu_str:<15} {ll_str:<15} {explanation}")


# 结果比较
print("\n模型比较汇总表:")
comparison_df = pd.DataFrame({
    'Copula': list(results.keys()),
    'Log-Likelihood': [results[k]['loglik'] for k in results.keys()],
    'AIC': [results[k]['AIC'] for k in results.keys()],
    'BIC': [results[k]['BIC'] for k in results.keys()],
    'Parameters': [results[k]['n_params'] for k in results.keys()],
    'Upper_Tail': [results[k]['upper_tail'] for k in results.keys()],
    'Lower_Tail': [results[k]['lower_tail'] for k in results.keys()]
})
comparison_df = comparison_df.sort_values('AIC')
print(comparison_df.to_string(index=False))

best_model = comparison_df.iloc[0]['Copula']
print(f"\n最佳模型(基于AIC): {best_model}")
print(f"最佳模型参数: {results[best_model]['params']}")
print(f"最佳模型上尾依赖 λ_U: {results[best_model]['upper_tail']:.6f}")
print(f"最佳模型下尾依赖 λ_L: {results[best_model]['lower_tail']:.6f}")

# 可视化部分
# 1. 生成网格数据
u_grid = np.linspace(0.01, 0.99, 100)
v_grid = np.linspace(0.01, 0.99, 100)
U_grid, V_grid = np.meshgrid(u_grid, v_grid)
grid_ravel = (U_grid.ravel(), V_grid.ravel())

# 2. 计算所有Copula的密度值
Z_gauss = gaussian_copula_pdf(*grid_ravel, rho_gauss).reshape(100, 100)
Z_t = student_t_copula_pdf(*grid_ravel, rho_t, nu_t).reshape(100, 100)
Z_clayton = clayton_copula_pdf(*grid_ravel, theta_clayton).reshape(100, 100)
Z_gumbel = gumbel_copula_pdf(*grid_ravel, theta_gumbel).reshape(100, 100)
Z_joe = joe_copula_pdf(*grid_ravel, theta_joe).reshape(100, 100)

# 3. 强制固定密度范围为0-5，超出范围的截断
vmin = 0
vmax = 5
Z_gauss = np.clip(Z_gauss, vmin, vmax)
Z_t = np.clip(Z_t, vmin, vmax)
Z_clayton = np.clip(Z_clayton, vmin, vmax)
Z_gumbel = np.clip(Z_gumbel, vmin, vmax)
Z_joe = np.clip(Z_joe, vmin, vmax)

levels = np.linspace(vmin, vmax, 20)  
norm = Normalize(vmin=vmin, vmax=vmax)

# 4. 全局样式配置
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

UNIFIED_SCATTER_COLOR = 'red'  # 散点 
CONTOUR_CMAP = 'turbo'            # 密度底色
CONTOUR_LINE_COLOR = 'black'       # 等高线轮廓线
CONTOUR_LINE_ALPHA = 0.9           # 轮廓线透明度

fig = plt.figure(figsize=(18, 12))

# ---------- 子图1：原始伪观测数据散点图 ----------
ax1 = plt.subplot(2, 3, 1)
plt.scatter(u, v, alpha=0.6, s=12, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.2)
plt.xlabel('U ', fontsize=11, fontweight='bold', fontfamily='SimSun')
plt.ylabel('V ', fontsize=11, fontweight='bold', fontfamily='SimSun')
plt.title('原始伪观测数据散点图', fontsize=12, fontweight='bold', pad=10, fontfamily='SimSun')
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
ax1.tick_params(width=1.2)
# 设置刻度字体为Times New Roman
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontname('Times New Roman')

# ---------- 子图2：Gaussian Copula ----------
ax2 = plt.subplot(2, 3, 2)
# 绘制密度填充（高对比度底色）
contour2 = plt.contourf(U_grid, V_grid, Z_gauss, levels=levels, cmap=CONTOUR_CMAP, alpha=0.9, norm=norm)
# 叠加黑色等高线轮廓（强化边界）
plt.contour(U_grid, V_grid, Z_gauss, levels=levels, colors=CONTOUR_LINE_COLOR, alpha=CONTOUR_LINE_ALPHA, linewidths=0.5)
# 统一深蓝色散点
plt.scatter(u, v, alpha=0.7, s=5, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.1)
plt.xlabel('U', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('V', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.title(f'Gaussian Copula\nρ={rho_gauss:.3f}, λ_U={results["Gaussian"]["upper_tail"]:.3f}, λ_L={results["Gaussian"]["lower_tail"]:.3f}', 
          fontsize=11, fontweight='bold', pad=8, fontfamily='Times New Roman')
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
ax2.tick_params(width=1.2)
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontname('Times New Roman')

# ---------- 子图3：Student-t Copula ----------
ax3 = plt.subplot(2, 3, 3)
plt.contourf(U_grid, V_grid, Z_t, levels=levels, cmap=CONTOUR_CMAP, alpha=0.9, norm=norm)
plt.contour(U_grid, V_grid, Z_t, levels=levels, colors=CONTOUR_LINE_COLOR, alpha=CONTOUR_LINE_ALPHA, linewidths=0.5)
plt.scatter(u, v, alpha=0.7, s=5, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.1)
plt.xlabel('U', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('V', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.title(f'Student-t Copula\nρ={rho_t:.3f}, ν={nu_t:.2f}\nλ_U={results["Student-t"]["upper_tail"]:.3f}, λ_L={results["Student-t"]["lower_tail"]:.3f}', 
          fontsize=11, fontweight='bold', pad=8, fontfamily='Times New Roman')
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
ax3.tick_params(width=1.2)
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontname('Times New Roman')

# ---------- 子图4：Clayton Copula ----------
ax4 = plt.subplot(2, 3, 4)
plt.contourf(U_grid, V_grid, Z_clayton, levels=levels, cmap=CONTOUR_CMAP, alpha=0.9, norm=norm)
plt.contour(U_grid, V_grid, Z_clayton, levels=levels, colors=CONTOUR_LINE_COLOR, alpha=CONTOUR_LINE_ALPHA, linewidths=0.5)
plt.scatter(u, v, alpha=0.7, s=5, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.1)
plt.xlabel('U', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('V', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.title(f'Clayton Copula\nθ={theta_clayton:.3f}\nλ_U={results["Clayton"]["upper_tail"]:.3f}, λ_L={results["Clayton"]["lower_tail"]:.3f}', 
          fontsize=11, fontweight='bold', pad=8, fontfamily='Times New Roman')
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
ax4.tick_params(width=1.2)
for label in ax4.get_xticklabels() + ax4.get_yticklabels():
    label.set_fontname('Times New Roman')

# ---------- 子图5：Gumbel Copula ----------
ax5 = plt.subplot(2, 3, 5)
plt.contourf(U_grid, V_grid, Z_gumbel, levels=levels, cmap=CONTOUR_CMAP, alpha=0.9, norm=norm)
plt.contour(U_grid, V_grid, Z_gumbel, levels=levels, colors=CONTOUR_LINE_COLOR, alpha=CONTOUR_LINE_ALPHA, linewidths=0.5)
plt.scatter(u, v, alpha=0.7, s=5, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.1)
plt.xlabel('U', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('V', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.title(f'Gumbel Copula\nθ={theta_gumbel:.3f}\nλ_U={results["Gumbel"]["upper_tail"]:.3f}, λ_L={results["Gumbel"]["lower_tail"]:.3f}', 
          fontsize=11, fontweight='bold', pad=8, fontfamily='Times New Roman')
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
ax5.tick_params(width=1.2)
for label in ax5.get_xticklabels() + ax5.get_yticklabels():
    label.set_fontname('Times New Roman')

# ---------- 子图6：Joe Copula ----------
ax6 = plt.subplot(2, 3, 6)
plt.contourf(U_grid, V_grid, Z_joe, levels=levels, cmap=CONTOUR_CMAP, alpha=0.9, norm=norm)
plt.contour(U_grid, V_grid, Z_joe, levels=levels, colors=CONTOUR_LINE_COLOR, alpha=CONTOUR_LINE_ALPHA, linewidths=0.5)
plt.scatter(u, v, alpha=0.7, s=5, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.1)
plt.xlabel('U', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('V', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.title(f'Joe Copula\nθ={theta_joe:.3f}\nλ_U={results["Joe"]["upper_tail"]:.3f}, λ_L={results["Joe"]["lower_tail"]:.3f}', 
          fontsize=11, fontweight='bold', pad=8, fontfamily='Times New Roman')
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
ax6.tick_params(width=1.2)
for label in ax6.get_xticklabels() + ax6.get_yticklabels():
    label.set_fontname('Times New Roman')

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(contour2, cax=cbar_ax, label='概率密度', extend='both')
cbar.set_label('概率密度', fontsize=12, fontweight='bold', fontfamily='SimSun')
cbar.ax.tick_params(width=1.2, labelsize=10)
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Times New Roman')

cbar_ticks = np.arange(vmin, vmax+1, 1)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f'{x:.1f}' for x in cbar_ticks])

plt.suptitle('Copula密度函数与尾部依赖特征', fontsize=14, fontweight='bold', y=0.99, fontfamily='SimSun')

# 调整布局
plt.tight_layout(rect=[0, 0, 0.9, 1])

# 保存高清图片
plt.savefig('copula_fitting_clear_contour.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

fig3 = plt.figure(figsize=(12, 6))

# 1. 尾部依赖系数柱状图
ax1 = plt.subplot(1, 2, 1)
models_list = list(results.keys())
upper_tails = [results[m]['upper_tail'] for m in models_list]
lower_tails = [results[m]['lower_tail'] for m in models_list]
x_pos = np.arange(len(models_list))
width = 0.35
bars1 = plt.bar(x_pos - width/2, upper_tails, width, label='上尾依赖 λ_U', 
                color='#e74c3c', edgecolor='black', linewidth=1.2)
bars2 = plt.bar(x_pos + width/2, lower_tails, width, label='下尾依赖 λ_L', 
                color='#3498db', edgecolor='black', linewidth=1.2)
plt.xlabel('Copula模型', fontsize=12, fontweight='bold', fontfamily='SimSun')
plt.ylabel('尾部依赖系数', fontsize=12, fontweight='bold', fontfamily='SimSun')
plt.title('尾部依赖系数对比', fontsize=13, fontweight='bold', fontfamily='SimSun')
plt.xticks(x_pos, models_list, rotation=45, ha='right', fontfamily='Times New Roman')
plt.legend(fontsize=11, loc='upper left', prop={'family':'SimSun', 'size':11})
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='显著阈值')
# 添加数值标签
for i, (u_val, l_val) in enumerate(zip(upper_tails, lower_tails)):
    if u_val > 0.005:
        plt.text(i - width/2, u_val + 0.01, f'{u_val:.3f}', ha='center', fontsize=9, fontweight='bold', fontfamily='Times New Roman')
    if l_val > 0.005:
        plt.text(i + width/2, l_val + 0.01, f'{l_val:.3f}', ha='center', fontsize=9, fontweight='bold', fontfamily='Times New Roman')

# 设置y轴刻度字体
for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')

# 2. 尾部依赖散点图
ax2 = plt.subplot(1, 2, 2)
colors_scatter = ['#2ecc71' if m == best_model else '#95a5a6' for m in models_list]
sizes = [400 if m == best_model else 200 for m in models_list]
plt.scatter(upper_tails, lower_tails, c=colors_scatter, s=sizes, 
            edgecolors='black', linewidth=2.5, alpha=0.8, zorder=3)
for i, model in enumerate(models_list):
    offset_x = 0.008 if model != best_model else 0.015
    offset_y = 0.008 if model != best_model else 0.015
    fontsize = 12 if model == best_model else 10
    fontweight = 'bold' if model == best_model else 'normal'
    plt.annotate(model, (upper_tails[i] + offset_x, lower_tails[i] + offset_y), 
                fontsize=fontsize, fontweight=fontweight, fontfamily='Times New Roman')
plt.xlabel('上尾依赖系数 λ_U', fontsize=12, fontweight='bold', fontfamily='SimSun')
plt.ylabel('下尾依赖系数 λ_L', fontsize=12, fontweight='bold', fontfamily='SimSun')
plt.title('尾部依赖结构图', fontsize=13, fontweight='bold', fontfamily='SimSun')
plt.grid(True, alpha=0.3, linestyle='--')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
plt.axvline(x=0.05, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)

# 添加象限标注
max_x = max(upper_tails) * 1.1
max_y = max(lower_tails) * 1.1
plt.text(max_x*0.7, max_y*0.9, '双尾依赖区', fontsize=10, alpha=0.6, fontfamily='SimSun',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
plt.text(max_x*0.7, max_y*0.1, '仅上尾依赖区', fontsize=10, alpha=0.6, fontfamily='SimSun',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
plt.text(max_x*0.1, max_y*0.9, '仅下尾依赖区', fontsize=10, alpha=0.6, fontfamily='SimSun',
         bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))

# 设置刻度字体
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontname('Times New Roman')

plt.suptitle('Copula尾部依赖特性分析', fontsize=14, fontweight='bold', y=0.98, fontfamily='SimSun')
plt.tight_layout()
plt.show()