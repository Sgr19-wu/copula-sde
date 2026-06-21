"""
完整显式时变参数非高斯状态依赖扩散SDE建模
================================================================================
模型: dX = θ(t)(μ(t) - X)dt + σ(t)|X|^γ dL_α

其中:
    θ(t) = c0 + c1·sin(ωt) + c2·cos(ωt)
    μ(t) = a0 + a1·sin(ωt) + a2·cos(ωt)
    σ(t) = b0 + b1·sin(ωt) + b2·cos(ωt)
    ω = 2π/365

参数估计：QMLE（Euler离散化近似）
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import kstest, norm, levy_stable
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

# 兼容acf_ljungbox
try:
    from statsmodels.stats.diagnostic import acf_ljungbox
except ImportError:
    def acf_ljungbox(x, lags=20, return_df=True):
        n = len(x)
        acf_vals = acf(x, nlags=lags, fft=False)
        q_stat = n * (n + 2) * np.sum(acf_vals[1:lags+1]**2 / (n - np.arange(1, lags+1)))
        p_value = 1 - stats.chi2.cdf(q_stat, lags)
        if return_df:
            return pd.DataFrame({'lb_pvalue': [p_value]}, index=[lags])
        return q_stat, p_value

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 0. 读取数据
# ============================================
df = pd.read_csv(r"D:\降水与径流量\data\residuals_1.csv")
res_up = df["upstream_residual"].dropna().values
res_down = df["downstream_residual"].dropna().values
dates = pd.to_datetime(df["date"].values)
dt = 1.0

print("=" * 80)
print("完整显式时变参数非高斯状态依赖扩散SDE建模")
print("=" * 80)
print(f"样本数: {len(res_up)}")
print(f"时间步长: dt = {dt} 天")
print(f"时间范围: {dates[0].date()} 到 {dates[-1].date()}")
print()

# ============================================
# 1. 数值稳定性常数
# ============================================
EPS = 1e-8
LOGPDF_MIN = -1e6
SCALE_MIN = 1e-4

# 时间索引
t_indices = np.arange(len(res_up))
PERIOD = 365
OMEGA = 2 * np.pi / PERIOD

# ============================================
# 2. 时变参数函数
# ============================================

def theta_t(t, c0, c1, c2):
    """时变均值回归速度"""
    return c0 + c1 * np.sin(OMEGA * t) + c2 * np.cos(OMEGA * t)

def mu_t(t, a0, a1, a2):
    """时变长期均值"""
    return a0 + a1 * np.sin(OMEGA * t) + a2 * np.cos(OMEGA * t)

def sigma_t(t, b0, b1, b2):
    """时变波动率（确保正值）"""
    val = b0 + b1 * np.sin(OMEGA * t) + b2 * np.cos(OMEGA * t)
    return np.maximum(val, SCALE_MIN)

# ============================================
# 3. 安全的α-stable对数密度
# ============================================

def safe_levy_stable_logpdf(x, alpha, beta):
    try:
        logpdf_val = levy_stable.logpdf(x, alpha, beta, 0, 1)
        if not np.isfinite(logpdf_val):
            return LOGPDF_MIN
        return max(logpdf_val, LOGPDF_MIN)
    except:
        return LOGPDF_MIN

# ============================================
# 4. 上游SDE参数估计（θ, μ, σ 全时变）
# ============================================
print("=" * 80)
print("【上游SDE参数估计（全时变）】")
print("模型: dX = θ(t)(μ(t) - X)dt + σ(t)|X|^γ dL_α")
print("θ(t) = c0 + c1·sin(ωt) + c2·cos(ωt)")
print("μ(t) = a0 + a1·sin(ωt) + a2·cos(ωt)")
print("σ(t) = b0 + b1·sin(ωt) + b2·cos(ωt)")
print("=" * 80)

def upstream_loglikelihood_full(params, data, dt, t_idx):
    """全时变参数SDE的负对数似然"""
    (c0, c1, c2,      # θ(t)参数
     a0, a1, a2,      # μ(t)参数
     b0, b1, b2,      # σ(t)参数
     gamma, alpha) = params
    beta = 0.0
    
    # 参数约束
    if c0 <= 0 or b0 <= 0 or gamma < 0:
        return 1e10
    if alpha <= 0.5 or alpha >= 2:
        return 1e10
    
    n = len(data)
    log_lik = 0
    
    for t in range(n-1):
        X_t = data[t]
        X_next = data[t+1]
        
        # 时变参数
        theta_val = theta_t(t_idx[t], c0, c1, c2)
        mu_val = mu_t(t_idx[t], a0, a1, a2)
        sigma_val = sigma_t(t_idx[t], b0, b1, b2)
        
        # 约束
        if theta_val <= 0:
            return 1e10
        
        # 漂移项
        cond_loc = X_t + theta_val * (mu_val - X_t) * dt
        
        # 扩散项
        power_term = max(abs(X_t) ** gamma, SCALE_MIN)
        cond_scale = sigma_val * power_term * (dt ** (1/alpha))
        
        if cond_scale < EPS:
            return 1e10
        
        std_resid = (X_next - cond_loc) / cond_scale
        logpdf_val = safe_levy_stable_logpdf(std_resid, alpha, beta)
        log_lik += logpdf_val - np.log(cond_scale)
        
        if log_lik < -1e10:
            return 1e10
    
    return -log_lik

# 初始值（基于之前估计）
init_up_full = [
    0.03, 0.01, 0.01,    # c0, c1, c2 (θ)
    -124, 100, 50,       # a0, a1, a2 (μ)
    21.4, 15, 6.5,       # b0, b1, b2 (σ)
    0.29,                # γ
    1.40                 # α
]

bounds_up_full = [
    (1e-6, 1), (-1, 1), (-1, 1),      # θ
    (-5000, 5000), (-1000, 1000), (-1000, 1000),  # μ
    (1e-6, 100), (-50, 50), (-50, 50),  # σ
    (0, 1.5),                         # γ
    (0.5, 2)                          # α
]

print("优化中（L-BFGS-B）...")
result_up_full = minimize(
    upstream_loglikelihood_full,
    init_up_full,
    args=(res_up, dt, t_indices),
    method='L-BFGS-B',
    bounds=bounds_up_full
)

params_up = result_up_full.x
(c0, c1, c2, a0, a1, a2, b0, b1, b2, gamma_up, alpha_up) = params_up
ll_up = -result_up_full.fun

print(f"\n估计结果:")
print(f"  θ(t) = {c0:.6f} + {c1:.6f}·sin(ωt) + {c2:.6f}·cos(ωt)")
print(f"  μ(t) = {a0:.2f} + {a1:.2f}·sin(ωt) + {a2:.2f}·cos(ωt)")
print(f"  σ(t) = {b0:.4f} + {b1:.4f}·sin(ωt) + {b2:.4f}·cos(ωt)")
print(f"  γ = {gamma_up:.6f}")
print(f"  α = {alpha_up:.6f}")
print(f"  LogLik = {ll_up:.2f}")
print()

# ============================================
# 5. 下游SDE参数估计（全时变，高斯）
# ============================================
print("=" * 80)
print("【下游SDE参数估计（全时变，高斯）】")
print("模型: dY = θ(t)(μ(t) - Y)dt + σ(t)|Y|^γ dW")
print("=" * 80)

def downstream_loglikelihood_full(params, data, dt, t_idx):
    """下游全时变SDE的负对数似然（高斯）"""
    (c0, c1, c2,      # θ(t)
     a0, a1, a2,      # μ(t)
     b0, b1, b2,      # σ(t)
     gamma) = params
    
    if c0 <= 0 or b0 <= 0 or gamma < 0:
        return 1e10
    
    n = len(data)
    log_lik = 0
    
    for t in range(n-1):
        Y_t = data[t]
        Y_next = data[t+1]
        
        theta_val = theta_t(t_idx[t], c0, c1, c2)
        mu_val = mu_t(t_idx[t], a0, a1, a2)
        sigma_val = sigma_t(t_idx[t], b0, b1, b2)
        
        if theta_val <= 0:
            return 1e10
        
        cond_loc = Y_t + theta_val * (mu_val - Y_t) * dt
        power_term = max(abs(Y_t) ** gamma, SCALE_MIN)
        cond_scale = sigma_val * power_term * np.sqrt(dt)
        
        if cond_scale < EPS:
            return 1e10
        
        log_lik += norm.logpdf(Y_next, loc=cond_loc, scale=cond_scale)
    
    return -log_lik

init_down_full = [
    0.20, 0.01, 0.01,   # c0, c1, c2
    -2030, 500, 300,    # a0, a1, a2
    17200, 2000, 1000,  # b0, b1, b2
    0.005               # γ
]

bounds_down_full = [
    (1e-6, 1), (-1, 1), (-1, 1),           # θ
    (-50000, 50000), (-10000, 10000), (-10000, 10000),  # μ
    (1e-6, 50000), (-10000, 10000), (-10000, 10000),    # σ
    (0, 1.5)                               # γ
]

print("优化中（L-BFGS-B）...")
result_down_full = minimize(
    downstream_loglikelihood_full,
    init_down_full,
    args=(res_down, dt, t_indices),
    method='L-BFGS-B',
    bounds=bounds_down_full
)

params_down = result_down_full.x
(c0_d, c1_d, c2_d, a0_d, a1_d, a2_d, b0_d, b1_d, b2_d, gamma_down) = params_down
ll_down = -result_down_full.fun

print(f"\n估计结果:")
print(f"  θ(t) = {c0_d:.6f} + {c1_d:.6f}·sin(ωt) + {c2_d:.6f}·cos(ωt)")
print(f"  μ(t) = {a0_d:.2f} + {a1_d:.2f}·sin(ωt) + {a2_d:.2f}·cos(ωt)")
print(f"  σ(t) = {b0_d:.2f} + {b1_d:.2f}·sin(ωt) + {b2_d:.2f}·cos(ωt)")
print(f"  γ = {gamma_down:.6f}")
print(f"  LogLik = {ll_down:.2f}")
print()

# ============================================
# 6. PIT变换
# ============================================
print("=" * 80)
print("【PIT变换】")
print("=" * 80)

# 上游PIT
u_up = np.zeros(len(res_up))
u_up[0] = 0.5
for t in range(1, len(res_up)):
    X_prev = res_up[t-1]
    
    theta_val = theta_t(t_indices[t-1], c0, c1, c2)
    mu_val = mu_t(t_indices[t-1], a0, a1, a2)
    sigma_val = sigma_t(t_indices[t-1], b0, b1, b2)
    
    cond_loc = X_prev + theta_val * (mu_val - X_prev) * dt
    power_term = max(abs(X_prev) ** gamma_up, SCALE_MIN)
    cond_scale = sigma_val * power_term * (dt ** (1/alpha_up))
    std_resid = (res_up[t] - cond_loc) / cond_scale
    u_up[t] = levy_stable.cdf(std_resid, alpha_up, 0.0, 0, 1)
    u_up[t] = np.clip(u_up[t], 1e-6, 1-1e-6)

# 下游PIT
u_down = np.zeros(len(res_down))
u_down[0] = 0.5
for t in range(1, len(res_down)):
    Y_prev = res_down[t-1]
    
    theta_val = theta_t(t_indices[t-1], c0_d, c1_d, c2_d)
    mu_val = mu_t(t_indices[t-1], a0_d, a1_d, a2_d)
    sigma_val = sigma_t(t_indices[t-1], b0_d, b1_d, b2_d)
    
    cond_loc = Y_prev + theta_val * (mu_val - Y_prev) * dt
    power_term = max(abs(Y_prev) ** gamma_down, SCALE_MIN)
    cond_scale = sigma_val * power_term * np.sqrt(dt)
    u_down[t] = norm.cdf(res_down[t], loc=cond_loc, scale=cond_scale)
    u_down[t] = np.clip(u_down[t], 1e-6, 1-1e-6)

print("  PIT变换完成")
print()

# ============================================
# 7. 模型诊断
# ============================================
print("=" * 80)
print("【模型诊断】")
print("=" * 80)

# 均匀性检验
ks_up = kstest(u_up, 'uniform')
ks_down = kstest(u_down, 'uniform')

# ACF分析
acf_up = acf(u_up - 0.5, nlags=20, fft=False)
acf_down = acf(u_down - 0.5, nlags=20, fft=False)

lag1_acf_up = acf_up[1] if len(acf_up) > 1 else 0
lag1_acf_down = acf_down[1] if len(acf_down) > 1 else 0
max_acf_up = np.max(np.abs(acf_up[1:])) if len(acf_up) > 1 else 0
max_acf_down = np.max(np.abs(acf_down[1:])) if len(acf_down) > 1 else 0

sig_threshold = 1.96 / np.sqrt(len(u_up))
sig_lags_up = np.sum(np.abs(acf_up[1:]) > sig_threshold) if len(acf_up) > 1 else 0
sig_lags_down = np.sum(np.abs(acf_down[1:]) > sig_threshold) if len(acf_down) > 1 else 0

# Ljung-Box检验
lb_up = acf_ljungbox(u_up - 0.5, lags=20, return_df=True)
lb_down = acf_ljungbox(u_down - 0.5, lags=20, return_df=True)

if hasattr(lb_up, 'iloc'):
    lb_pvalues_up = lb_up['lb_pvalue'].values
    lb_pvalues_down = lb_down['lb_pvalue'].values
else:
    lb_pvalues_up = np.array(lb_up)
    lb_pvalues_down = np.array(lb_down)

# 通过率
pass_rate_up_lag5 = np.mean(lb_pvalues_up[:5] > 0.05)
pass_rate_up_lag10 = np.mean(lb_pvalues_up[:10] > 0.05)
pass_rate_up_all = np.mean(lb_pvalues_up > 0.05)

pass_rate_down_lag5 = np.mean(lb_pvalues_down[:5] > 0.05)
pass_rate_down_lag10 = np.mean(lb_pvalues_down[:10] > 0.05)
pass_rate_down_all = np.mean(lb_pvalues_down > 0.05)

min_p_up = lb_pvalues_up.min()
min_p_down = lb_pvalues_down.min()

# 信息准则
n_up = len(res_up)
n_down = len(res_down)
k_up = 11  # c0,c1,c2, a0,a1,a2, b0,b1,b2, γ, α
k_down = 10 # c0,c1,c2, a0,a1,a2, b0,b1,b2, γ

aic_up = 2 * k_up - 2 * ll_up
bic_up = k_up * np.log(n_up) - 2 * ll_up
aic_down = 2 * k_down - 2 * ll_down
bic_down = k_down * np.log(n_down) - 2 * ll_down

print("\n【上游模型诊断】")
print("-" * 50)
print(f"  KS检验: 统计量={ks_up.statistic:.6f}, p值={ks_up.pvalue:.6f}")
print(f"  结论: {'✅ 通过' if ks_up.statistic < 0.05 else '❌ 不通过'}")
print()
print(f"  自相关分析 (ACF):")
print(f"    滞后1 ACF = {lag1_acf_up:.6f}")
print(f"    最大|ACF| (滞后1-20) = {max_acf_up:.6f}")
print(f"    显著滞后数 = {sig_lags_up}/20")
print()
print(f"  Ljung-Box检验:")
print(f"    前5阶通过率: {pass_rate_up_lag5*100:.1f}% ({int(pass_rate_up_lag5*5)}/5)")
print(f"    前10阶通过率: {pass_rate_up_lag10*100:.1f}% ({int(pass_rate_up_lag10*10)}/10)")
print(f"    全部20阶通过率: {pass_rate_up_all*100:.1f}% ({int(pass_rate_up_all*20)}/20)")
print(f"    最小p值: {min_p_up:.6f}")
print()
print(f"  信息准则:")
print(f"    AIC = {aic_up:.2f}, BIC = {bic_up:.2f}")

print("\n【下游模型诊断】")
print("-" * 50)
print(f"  KS检验: 统计量={ks_down.statistic:.6f}, p值={ks_down.pvalue:.6f}")
print(f"  结论: {'✅ 通过' if ks_down.statistic < 0.05 else '❌ 不通过'}")
print()
print(f"  自相关分析 (ACF):")
print(f"    滞后1 ACF = {lag1_acf_down:.6f}")
print(f"    最大|ACF| (滞后1-20) = {max_acf_down:.6f}")
print(f"    显著滞后数 = {sig_lags_down}/20")
print()
print(f"  Ljung-Box检验:")
print(f"    前5阶通过率: {pass_rate_down_lag5*100:.1f}% ({int(pass_rate_down_lag5*5)}/5)")
print(f"    前10阶通过率: {pass_rate_down_lag10*100:.1f}% ({int(pass_rate_down_lag10*10)}/10)")
print(f"    全部20阶通过率: {pass_rate_down_all*100:.1f}% ({int(pass_rate_down_all*20)}/20)")
print(f"    最小p值: {min_p_down:.6f}")
print()
print(f"  信息准则:")
print(f"    AIC = {aic_down:.2f}, BIC = {bic_down:.2f}")

# ============================================
# 8. 可视化
# ============================================
fig, axes = plt.subplots(2, 4, figsize=(16, 10))

# 计算时变参数序列用于绘图
t_plot = np.arange(len(res_up))
theta_up_plot = theta_t(t_plot, c0, c1, c2)
mu_up_plot = mu_t(t_plot, a0, a1, a2)
sigma_up_plot = sigma_t(t_plot, b0, b1, b2)
theta_down_plot = theta_t(t_plot, c0_d, c1_d, c2_d)
mu_down_plot = mu_t(t_plot, a0_d, a1_d, a2_d)
sigma_down_plot = sigma_t(t_plot, b0_d, b1_d, b2_d)

# 第一行：上游
axes[0, 0].plot(dates, theta_up_plot, 'b-', linewidth=1)
axes[0, 0].set_title('上游 θ(t) (时变回归速度)')
axes[0, 0].set_xlabel('时间')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(dates, mu_up_plot, 'b-', linewidth=1)
axes[0, 1].set_title('上游 μ(t) (时变长期均值)')
axes[0, 1].set_xlabel('时间')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(dates, sigma_up_plot, 'b-', linewidth=1)
axes[0, 2].set_title('上游 σ(t) (时变波动率)')
axes[0, 2].set_xlabel('时间')
axes[0, 2].grid(True, alpha=0.3)

axes[0, 3].hist(u_up, bins=50, density=True, range=(0,1), alpha=0.7, edgecolor='black')
axes[0, 3].axhline(y=1, color='r', linestyle='--')
axes[0, 3].set_title(f'上游PIT (KS={ks_up.statistic:.4f})')
axes[0, 3].set_xlabel('u')

# 第二行：下游
axes[1, 0].plot(dates, theta_down_plot, 'r-', linewidth=1)
axes[1, 0].set_title('下游 θ(t) (时变回归速度)')
axes[1, 0].set_xlabel('时间')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(dates, mu_down_plot, 'r-', linewidth=1)
axes[1, 1].set_title('下游 μ(t) (时变长期均值)')
axes[1, 1].set_xlabel('时间')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(dates, sigma_down_plot, 'r-', linewidth=1)
axes[1, 2].set_title('下游 σ(t) (时变波动率)')
axes[1, 2].set_xlabel('时间')
axes[1, 2].grid(True, alpha=0.3)

axes[1, 3].hist(u_down, bins=50, density=True, range=(0,1), alpha=0.7, edgecolor='black')
axes[1, 3].axhline(y=1, color='r', linestyle='--')
axes[1, 3].set_title(f'下游PIT (KS={ks_down.statistic:.4f})')
axes[1, 3].set_xlabel('u')

plt.tight_layout()
plt.savefig('full_harmonic_sde_results.png', dpi=150)
plt.show()

# ============================================
# 9. 保存结果
# ============================================

pit_df = pd.DataFrame({
    'date': dates,
    'upstream_pit': u_up,
    'downstream_pit': u_down
})
pit_df.to_csv('pit_series_full_harmonic.csv', index=False)
print("\n✓ PIT序列已保存")

print("\n" + "=" * 80)
print("建模完成！")
print("=" * 80)