import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import levy_stable, t, norm
from scipy.optimize import minimize

# ===================== 核心字体配置（中英分离） =====================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']  # 宋体优先（中文）
plt.rcParams['font.sans-serif'] = ['Times New Roman','SimSun',]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300  # 高清输出
plt.rcParams['savefig.dpi'] = 300

# ===================== 加载数据 + 拟合分布参数（绘图依赖） =====================
# 1. 加载残差数据
RES_PATH = r"D:\降水与径流量\data\residuals_1.csv"
df = pd.read_csv(RES_PATH)
up = df["upstream_residual"].dropna().values  # 过滤空值避免报错
down = df["downstream_residual"].dropna().values

# 2. 上游 α-stable 分布拟合（绘图依赖参数）
alpha0 = 1.3
beta0 = 0.1
loc0 = np.mean(up)
scale0 = np.std(up)
params_up = levy_stable.fit(up, alpha0, beta0, loc=loc0, scale=scale0)
alpha_up, beta_up, delta_up, gamma_up = params_up

# 3. 下游 t 分布拟合（绘图依赖参数）
def neg_loglik_t(params, x):
    df, loc, scale = params
    if df <= 1 or scale <= 0:
        return np.inf
    return -np.sum(t.logpdf(x, df=df, loc=loc, scale=scale))

init = [4, np.mean(down), np.std(down)]
bounds = [(1.1, 50), (None, None), (1e-6, None)]
res = minimize(neg_loglik_t, init, args=(down,), bounds=bounds)
df_down, loc_down, scale_down = res.x


def plot_fit(name, data, ys_func, label, filename):
    plt.figure(figsize=(8, 5))
    # 绘制样本直方图
    plt.hist(data, bins=100, density=True, alpha=0.5, color='steelblue', label='样本分布')

    # 生成横坐标
    xs = np.linspace(np.min(data), np.max(data), 400)
    # 绘制目标拟合分布曲线（α-stable/t）
    plt.plot(xs, ys_func(xs), 'r-', lw=2, label=label)
    # 绘制正态分布对比曲线
    normal_mean = np.mean(data)
    normal_std = np.std(data)
    plt.plot(xs, norm.pdf(xs, loc=normal_mean, scale=normal_std), 
             'g--', lw=2, label='正态分布')

    # 设置标题/标签
    plt.title(name, fontsize=12, fontweight='bold', fontfamily='SimSun')
    plt.xlabel("残差", fontsize=11, fontfamily='SimSun')  
    plt.ylabel("概率密度", fontsize=11, fontfamily='SimSun')  
    
    # 获取图例句柄和标签
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # 创建图例，不使用prop参数
    legend = plt.legend(handles=handles, loc='upper right', frameon=True, 
                       framealpha=0.9, edgecolor='none')
    
    # 简单粗暴的方法：所有包含中文的用宋体，其余用Times New Roman
    for text in legend.get_texts():
        if any(c in text.get_text() for c in ['样本', '正态']):
            text.set_fontname('SimSun')
        else:
            text.set_fontname('Times New Roman')
    
    plt.grid(alpha=0.3)
    
    # 刻度字体
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# 绘制上游 α-stable 拟合曲线
plot_fit(
    "上游残差分布及拟合曲线",
    up,
    lambda x: levy_stable.pdf(x, alpha_up, beta_up, loc=delta_up, scale=gamma_up),
    f"α-stable(α={alpha_up:.2f}, β={beta_up:.2f})",
    "upstream_stable_fit.png"
)

# 绘制下游 t 分布拟合曲线
plot_fit(
    "下游残差分布及拟合曲线",
    down,
    lambda x: t.pdf(x, df_down, loc=loc_down, scale=scale_down),
    f"t分布(df={df_down:.2f})",
    "downstream_t_fit.png"
)