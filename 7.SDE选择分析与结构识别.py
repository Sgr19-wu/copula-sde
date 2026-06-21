import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller

# 全局设置新罗马字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 0 参数
dt = 1.0   # 日尺度

# 1 读取数据
df = pd.read_csv(r"D:\降水与径流量\data\residuals_1.csv")
df["date"] = pd.to_datetime(df["date"])

res_up = df["upstream_residual"].dropna().values
res_down = df["downstream_residual"].dropna().values
time = df["date"].values

# 2 增量 ΔX
def compute_increment(x,dt):
    return np.diff(x) / dt

d_up = compute_increment(res_up,dt)
d_down = compute_increment(res_down,dt)

print("Upstream increment mean:",np.mean(d_up))
print("Upstream increment std:",np.std(d_up))
print("Downstream increment mean:",np.mean(d_down))
print("Downstream increment std:",np.std(d_down))

# 3 平稳性检验（ADF）
def adf_test(x,name):
    result = adfuller(x)
    print(f"\nADF Test - {name}")
    print("ADF Statistic:",result[0])
    print("p-value:",result[1])

adf_test(res_up,"Upstream")
adf_test(res_down,"Downstream")

# 4 条件漂移 E[dX|X]
def conditional_drift(x,dx,bins=20):
    x_t = x[:-1]
    edges = np.quantile(x_t, np.linspace(0,1,bins+1))
    x_mid=[]
    drift=[]
    for i in range(bins):
        idx=(x_t>=edges[i])&(x_t<edges[i+1])
        if np.sum(idx)>30:
            x_mid.append(np.mean(x_t[idx]))
            drift.append(np.mean(dx[idx]))
    return np.array(x_mid),np.array(drift)

x_up,drift_up = conditional_drift(res_up,d_up)
x_down,drift_down = conditional_drift(res_down,d_down)

plt.figure()
plt.scatter(x_up,drift_up)
plt.title("Conditional Drift (Upstream)")
plt.xlabel("X")
plt.ylabel("E[dX|X]")
plt.show()

plt.figure()
plt.scatter(x_down,drift_down)
plt.title("Conditional Drift (Downstream)")
plt.xlabel("X")
plt.ylabel("E[dX|X]")
plt.show()

# 5 条件扩散 Var[dX|X] - 核心修改部分
def conditional_diffusion(x,dx,bins=20):
    x_t = x[:-1]
    edges = np.quantile(x_t, np.linspace(0,1,bins+1))
    x_mid=[]
    diffusion=[]
    for i in range(bins):
        idx=(x_t>=edges[i])&(x_t<edges[i+1])
        if np.sum(idx)>30:
            # 关键修改：去掉 np.abs()，使用原始X值的均值
            x_mid.append(np.mean(x_t[idx]))  
            diffusion.append(np.var(dx[idx]))
    return np.array(x_mid),np.array(diffusion)

x_up2,diff_up = conditional_diffusion(res_up,d_up)
x_down2,diff_down = conditional_diffusion(res_down,d_down)

plt.figure()
plt.scatter(x_up2,diff_up)
plt.title("Conditional Diffusion (Upstream)")
plt.xlabel("X")  # 修改横坐标标签为 X（去掉 |X|）
plt.ylabel("Var[dX|X]")
plt.show()

plt.figure()
plt.scatter(x_down2,diff_down)
plt.title("Conditional Diffusion (Downstream)")
plt.xlabel("X")  # 修改横坐标标签为 X（去掉 |X|）
plt.ylabel("Var[dX|X]")
plt.show()

# 6 增量分布分析
def increment_analysis(dx,title):
    plt.figure()
    plt.hist(dx,bins=100,density=True)
    plt.title(title+" Increment Histogram")
    plt.show()

    qqplot(dx,line='s')
    plt.title(title+" QQ plot")
    plt.show()

    print(f"\nNormality test ({title})")
    print("Jarque-Bera:",stats.jarque_bera(dx))

increment_analysis(d_up,"Upstream")
increment_analysis(d_down,"Downstream")

# 7 极端事件分析
def tail_analysis(x,title):
    x_sorted=np.sort(np.abs(x))
    ccdf=1-np.arange(1,len(x_sorted)+1)/len(x_sorted)
    plt.figure()
    plt.loglog(x_sorted,ccdf)
    plt.title(title+" Tail (log-log)")
    plt.xlabel("|X|")
    plt.ylabel("P(X>x)")
    plt.show()

tail_analysis(res_up,"Upstream")
tail_analysis(res_down,"Downstream")

# 8 跳跃检测
def jump_detection(dx,title):
    std=np.std(dx)
    threshold=3*std
    jump_idx=np.where(np.abs(dx)>threshold)[0]
    print(f"\n{title} Jump Detection")
    print("Threshold:",threshold)
    print("Count:",len(jump_idx))
    print("Proportion:",len(jump_idx)/len(dx))

jump_detection(d_up,"Upstream")
jump_detection(d_down,"Downstream")

# 9 自相关结构
plt.figure()
plot_acf(d_up,lags=50)
plt.title("Increment ACF Upstream")
plt.show()

plt.figure()
plot_acf(d_down,lags=50)
plt.title("Increment ACF Downstream")
plt.show()

# 10 时间序列图
plt.figure(figsize=(12,4))
plt.plot(time,res_up)
plt.title("Upstream Residual")
plt.show()

plt.figure(figsize=(12,4))
plt.plot(time,res_down)
plt.title("Downstream Residual")
plt.show()