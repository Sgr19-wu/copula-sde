import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera, shapiro, anderson
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei','Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果保存目录
if not os.path.exists('analysis_results'):
    os.makedirs('analysis_results')

class ColumbiaDataAnalyzer:
    def __init__(self, data_path=None):
        self.data = pd.read_csv(data_path) if data_path else None
        self.processed_data = None
        self.summary_stats = {}
        
    def basic_info(self):
        # 修正：先临时解析日期，确保时间范围显示准确
        temp_data = self.data.copy()
        temp_data['date_parsed'] = pd.to_datetime(temp_data['date'], errors='coerce')
        
        print(f"数据维度: {self.data.shape}")
        # 使用解析后的日期计算时间范围（不再用字符串比较）
        print(f"时间范围: {temp_data['date_parsed'].min()} 至 {temp_data['date_parsed'].max()}")
        # 计算理论天数和缺失日期数
        total_days = (temp_data['date_parsed'].max() - temp_data['date_parsed'].min()).days + 1
        missing_days = total_days - len(temp_data['date_parsed'].dropna())
        print(f"理论应有的天数: {total_days}")
        print(f"实际数据行数: {len(self.data)}")
        print(f"时间范围内缺失的日期数量: {missing_days}")
        
        print(f"\n列名及数据类型:\n{self.data.dtypes}")
        
        missing_stats = self.data.isnull().sum()
        missing_percent = (missing_stats / len(self.data)) * 100
        missing_df = pd.DataFrame({'缺失值数量': missing_stats, '缺失率(%)': missing_percent.round(4)})
        print(f"\n缺失值统计:\n{missing_df}")
        
        # 保存基础信息到CSV
        missing_df.to_csv('analysis_results/01_基础信息_缺失值统计.csv', encoding='utf-8-sig')
        # 保存数据维度和时间范围信息
        basic_info_dict = {
            '数据行数': [self.data.shape[0]],
            '数据列数': [self.data.shape[1]],
            '时间起始': [temp_data['date_parsed'].min()],
            '时间结束': [temp_data['date_parsed'].max()],
            '理论天数': [total_days],
            '缺失日期数': [missing_days]
        }
        pd.DataFrame(basic_info_dict).to_csv('analysis_results/01_基础信息_数据维度与时间范围.csv', 
                                            index=False, encoding='utf-8-sig')
        
        return missing_df
    
    def data_preprocessing(self):
        
        self.processed_data = self.data.copy()
        
        # 修正：显式指定日期解析格式（增强鲁棒性），处理解析错误
        self.processed_data['date'] = pd.to_datetime(self.processed_data['date'], 
                                                    format='%Y/%m/%d',  # 根据你的日期格式调整，若不确定可去掉该参数
                                                    errors='coerce')
        # 删除解析失败的行（如果有）
        parse_fail_count = self.processed_data['date'].isnull().sum()
        if parse_fail_count > 0:
            print(f"\n日期解析失败的记录数: {parse_fail_count}，已删除")
            self.processed_data = self.processed_data.dropna(subset=['date'])
        
        self.processed_data = self.processed_data.sort_values('date').reset_index(drop=True)
        
        # 时间特征
        self.processed_data['year'] = self.processed_data['date'].dt.year
        self.processed_data['month'] = self.processed_data['date'].dt.month
        self.processed_data['day_of_year'] = self.processed_data['date'].dt.dayofyear
        # 修改季节映射为春夏秋冬顺序
        season_map = {3: 'Spring', 4: 'Spring', 5: 'Spring',  # 春季：3-5月
                      6: 'Summer', 7: 'Summer', 8: 'Summer',  # 夏季：6-8月
                      9: 'Autumn', 10: 'Autumn', 11: 'Autumn',  # 秋季：9-11月
                      12: 'Winter', 1: 'Winter', 2: 'Winter'}   # 冬季：12-2月
        self.processed_data['season'] = self.processed_data['month'].map(season_map)
        
        # 缺失值处理
        numeric_cols = ['precipitation', 'temperature', 'swe', 'discharge_upstream', 'discharge_downstream']
        print("缺失值处理:")
        missing_info = {}
        for col in numeric_cols:
            missing_count = self.processed_data[col].isnull().sum()
            missing_info[col] = missing_count
            if missing_count > 0:
                print(f"  {col}: {missing_count} 个缺失值 -> 线性插值填补")
                self.processed_data[col] = self.processed_data[col].interpolate(method='linear')
        
        # 保存缺失值处理信息
        missing_df = pd.DataFrame(list(missing_info.items()), 
                                columns=['变量名', '缺失值数量'])
        missing_df['处理方式'] = np.where(missing_df['缺失值数量'] > 0, '线性插值填补', '无缺失值')
        missing_df.to_csv('analysis_results/02_预处理_缺失值处理.csv', index=False, encoding='utf-8-sig')
        
        # 异常值检测
        print("\n异常值检测 (IQR)")
        outlier_summary = {}
        for col in numeric_cols:
            Q1, Q3 = self.processed_data[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = ((self.processed_data[col] < lower_bound) | (self.processed_data[col] > upper_bound))
            outlier_count = outliers.sum()
            outlier_summary[col] = {
                'count': outlier_count, 
                'percentage': (outlier_count / len(self.processed_data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            print(f"  {col}: {outlier_count} 个统计异常值 ({outlier_summary[col]['percentage']:.2f}%) ")
        
        # 保存异常值检测结果
        outlier_df = pd.DataFrame.from_dict(outlier_summary, orient='index')
        outlier_df = outlier_df.round(4)
        outlier_df.to_csv('analysis_results/02_预处理_异常值检测.csv', encoding='utf-8-sig')
        
        # 保存预处理后的数据
        self.processed_data.to_csv('analysis_results/02_预处理后完整数据.csv', 
                                  index=False, encoding='utf-8-sig')
        
        print(f"\n预处理完成！处理后数据维度: {self.processed_data.shape}")
        return self.processed_data, outlier_summary
    
    def descriptive_statistics(self):
        numeric_cols = ['precipitation', 'temperature', 'swe', 'discharge_upstream', 'discharge_downstream']
        
        # 基本统计量
        desc_stats = self.processed_data[numeric_cols].describe()
        skewness = self.processed_data[numeric_cols].skew()
        kurtosis = self.processed_data[numeric_cols].kurtosis()
        
        extended_stats = desc_stats.copy()
        extended_stats.loc['skewness'] = skewness
        extended_stats.loc['kurtosis'] = kurtosis
        print("描述性统计:\n", extended_stats.round(4))
        
        # 保存描述性统计结果
        extended_stats.round(4).to_csv('analysis_results/03_描述性统计_基本统计量.csv', encoding='utf-8-sig')
        
        # 分布检验
        print("\n分布特征分析:")
        distribution_tests = {}
        test_results = []
        
        for col in numeric_cols:
            data_col = self.processed_data[col].dropna()
            jb_stat, jb_pvalue = jarque_bera(data_col)
            
            if len(data_col) <= 5000:
                sw_stat, sw_pvalue = shapiro(data_col[:5000])
            else:
                sw_stat, sw_pvalue = np.nan, np.nan
            
            ad_result = anderson(data_col, dist='norm')
            ad_stat, ad_critical = ad_result.statistic, ad_result.critical_values[2]
            
            distribution_tests[col] = {
                'JB_statistic': jb_stat, 'JB_pvalue': jb_pvalue,
                'SW_statistic': sw_stat, 'SW_pvalue': sw_pvalue,
                'AD_statistic': ad_stat, 'AD_critical_5%': ad_critical,
                'is_normal_JB': jb_pvalue > 0.05,
                'is_normal_SW': sw_pvalue > 0.05 if not np.isnan(sw_pvalue) else None,
                'is_normal_AD': ad_stat < ad_critical
            }
            
            # 整理检验结果用于保存
            skew_desc = '右偏' if skewness[col] > 0 else '左偏' if skewness[col] < 0 else '对称'
            kurt_desc = '尖峭' if kurtosis[col] > 0 else '平坦' if kurtosis[col] < 0 else '正常'
            normal_desc = '正态' if jb_pvalue > 0.05 else '非正态'
            
            test_results.append({
                '变量名': col,
                '偏度': round(skewness[col], 4),
                '偏度描述': skew_desc,
                '峰度': round(kurtosis[col], 4),
                '峰度描述': kurt_desc,
                'JB统计量': round(jb_stat, 4),
                'JB_p值': round(jb_pvalue, 6),
                'JB检验结果': normal_desc,
                'SW统计量': round(sw_stat, 4) if not np.isnan(sw_stat) else np.nan,
                'SW_p值': round(sw_pvalue, 6) if not np.isnan(sw_pvalue) else np.nan,
                'AD统计量': round(ad_stat, 4),
                'AD临界值(5%)': round(ad_critical, 4),
                'AD检验结果': '正态' if ad_stat < ad_critical else '非正态'
            })
            
            print(f"\n{col}:")
            print(f"  偏度: {skewness[col]:.4f} ({skew_desc})")
            print(f"  峰度: {kurtosis[col]:.4f} ({kurt_desc})")
            print(f"  JB检验: p值={jb_pvalue:.6f} ({normal_desc})")
            print(f"  AD检验: 统计量={ad_stat:.4f}, 临界值={ad_critical:.4f}")
        
        # 保存分布检验结果
        test_df = pd.DataFrame(test_results)
        test_df.to_csv('analysis_results/03_描述性统计_分布检验.csv', index=False, encoding='utf-8-sig')
        
        self.summary_stats['descriptive'] = extended_stats
        self.summary_stats['distribution_tests'] = distribution_tests
        return extended_stats, distribution_tests
    
    def correlation_analysis(self):
        print("相关性与依赖结构分析")
        
        numeric_cols = ['precipitation', 'temperature', 'swe', 'discharge_upstream', 'discharge_downstream']
        
        pearson_corr = self.processed_data[numeric_cols].corr()
        spearman_corr = self.processed_data[numeric_cols].corr(method='spearman')
        kendall_corr = self.processed_data[numeric_cols].corr(method='kendall')
        
        print("Pearson线性相关系数矩阵:\n", pearson_corr.round(4))
        print("\nSpearman秩相关系数矩阵:\n", spearman_corr.round(4))
        print("\nKendall's τ相关系数矩阵:\n", kendall_corr.round(4))
        
        # 保存相关性矩阵
        pearson_corr.round(4).to_csv('analysis_results/04_相关性分析_Pearson系数.csv', encoding='utf-8-sig')
        spearman_corr.round(4).to_csv('analysis_results/04_相关性分析_Spearman系数.csv', encoding='utf-8-sig')
        kendall_corr.round(4).to_csv('analysis_results/04_相关性分析_Kendall系数.csv', encoding='utf-8-sig')
        
        upstream_downstream_corr = {
            'Pearson': pearson_corr.loc['discharge_upstream', 'discharge_downstream'],
            'Spearman': spearman_corr.loc['discharge_upstream', 'discharge_downstream'],
            'Kendall': kendall_corr.loc['discharge_upstream', 'discharge_downstream']
        }
        
        # 保存上下游径流相关性
        ud_corr_df = pd.DataFrame(list(upstream_downstream_corr.items()), 
                                columns=['相关系数类型', '上下游径流相关系数'])
        ud_corr_df['相关系数'] = ud_corr_df['上下游径流相关系数'].round(4)
        ud_corr_df['相关性强度'] = np.where(ud_corr_df['相关系数'] > 0.8, '强',
                                        np.where(ud_corr_df['相关系数'] > 0.5, '中等', '弱'))
        ud_corr_df.to_csv('analysis_results/04_相关性分析_上下游径流相关性.csv', 
                         index=False, encoding='utf-8-sig')
        
        print(f"\n上下游径流量相关性:")
        for method, corr_val in upstream_downstream_corr.items():
            print(f"  {method}: {corr_val:.4f}")
        
        self.summary_stats['correlations'] = {
            'pearson': pearson_corr, 'spearman': spearman_corr, 
            'kendall': kendall_corr, 'upstream_downstream': upstream_downstream_corr
        }
        return pearson_corr, spearman_corr, kendall_corr
    
    def seasonal_analysis(self):
        numeric_cols = ['precipitation', 'temperature', 'swe', 'discharge_upstream', 'discharge_downstream']
        seasonal_stats = self.processed_data.groupby('season')[numeric_cols].agg(['mean', 'std', 'min', 'max'])
        monthly_stats = self.processed_data.groupby('month')[numeric_cols].mean()
        
        print("季节性统计:\n", seasonal_stats.round(2))
        print(f"\n月度平均值:\n", monthly_stats.round(2))
        
        # 保存季节性统计结果
        seasonal_stats.round(2).to_csv('analysis_results/05_季节性分析_季节统计.csv', encoding='utf-8-sig')
        monthly_stats.round(2).to_csv('analysis_results/05_季节性分析_月度平均值.csv', encoding='utf-8-sig')
        
        self.summary_stats['seasonal'] = {'by_season': seasonal_stats, 'by_month': monthly_stats}
        return seasonal_stats, monthly_stats
    
    def extreme_value_analysis(self):
        numeric_cols = ['precipitation', 'temperature', 'swe', 'discharge_upstream', 'discharge_downstream']
        extreme_stats = {}
        extreme_results = []
        
        for col in numeric_cols:
            data_col = self.processed_data[col]
            upper_threshold = float(data_col.quantile(0.95))
            lower_threshold = float(data_col.quantile(0.05))
            high_extremes = data_col >= upper_threshold
            low_extremes = data_col <= lower_threshold
            
            extreme_stats[col] = {
                'upper_threshold_95%': upper_threshold, 'lower_threshold_5%': lower_threshold,
                'high_extreme_count': high_extremes.sum(), 'low_extreme_count': low_extremes.sum()
            }
            
            # 整理极值分析结果
            extreme_results.append({
                '变量名': col,
                '上极值阈值(95%)': round(upper_threshold, 2),
                '下极值阈值(5%)': round(lower_threshold, 2),
                '高极值事件数': high_extremes.sum(),
                '低极值事件数': low_extremes.sum(),
                '高极值占比(%)': round(high_extremes.sum()/len(self.processed_data)*100, 2),
                '低极值占比(%)': round(low_extremes.sum()/len(self.processed_data)*100, 2)
            })
            
            print(f"\n{col}:")
            print(f"  上极值阈值(95%): {upper_threshold:.2f}")
            print(f"  下极值阈值(5%): {lower_threshold:.2f}")
            print(f"  高极值事件: {high_extremes.sum()} 次")
            print(f"  低极值事件: {low_extremes.sum()} 次")
        
        # 联合极值分析
        upstream_high = self.processed_data['discharge_upstream'] >= self.processed_data['discharge_upstream'].quantile(0.95)
        downstream_high = self.processed_data['discharge_downstream'] >= self.processed_data['discharge_downstream'].quantile(0.95)
        joint_high_extremes = upstream_high & downstream_high
        
        joint_extreme = {
            '上游高径流事件数': upstream_high.sum(),
            '下游高径流事件数': downstream_high.sum(),
            '联合高径流事件数': joint_high_extremes.sum(),
            '联合发生概率(%)': round(joint_high_extremes.sum() / len(self.processed_data) * 100, 2)
        }
        
        # 保存极值分析结果
        extreme_df = pd.DataFrame(extreme_results)
        extreme_df.to_csv('analysis_results/06_极值分析_单变量极值.csv', index=False, encoding='utf-8-sig')
        
        joint_df = pd.DataFrame([joint_extreme])
        joint_df.to_csv('analysis_results/06_极值分析_联合极值.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n联合极值分析:")
        print(f"  上游高径流事件: {upstream_high.sum()} 次")
        print(f"  下游高径流事件: {downstream_high.sum()} 次")
        print(f"  联合高径流事件: {joint_high_extremes.sum()} 次")
        print(f"  联合发生概率: {joint_high_extremes.sum() / len(self.processed_data) * 100:.2f}%")
        
        self.summary_stats['extremes'] = extreme_stats
        return extreme_stats
    
    def generate_visualizations(self):
        numeric_cols = ['precipitation', 'temperature', 'swe', 'discharge_upstream', 'discharge_downstream']
        
        # 时间序列图
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(15, 20))
        for i, col in enumerate(numeric_cols):
            axes[i].plot(self.processed_data['date'], self.processed_data[col], alpha=0.7)
            axes[i].set_title(f'{col} 时间序列', fontsize=12)
            axes[i].set_xlabel('日期')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analysis_results/可视化_时间序列图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 分布直方图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            axes[i].hist(self.processed_data[col], bins=50, alpha=0.7, density=True)
            axes[i].set_title(f'{col} 分布直方图', fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('密度')
            axes[i].grid(True, alpha=0.3)
        axes[-1].axis('off')
        plt.tight_layout()
        plt.savefig('analysis_results/可视化_分布直方图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 相关性热图
        plt.figure(figsize=(10, 8))
        pearson_corr = self.processed_data[numeric_cols].corr()
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('变量间Pearson相关系数热图', fontsize=14)
        plt.tight_layout()
        plt.savefig('analysis_results/可视化_相关性热图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 上下游径流量散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(self.processed_data['discharge_upstream'], 
                   self.processed_data['discharge_downstream'], alpha=0.6, s=10)
        plt.xlabel('上游径流量 (CFS)', fontsize=12)
        plt.ylabel('下游径流量 (CFS)', fontsize=12)
        plt.title('上下游径流量关系散点图', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加回归线
        z = np.polyfit(self.processed_data['discharge_upstream'], 
                      self.processed_data['discharge_downstream'], 1)
        p = np.poly1d(z)
        plt.plot(self.processed_data['discharge_upstream'], 
                p(self.processed_data['discharge_upstream']), "r--", alpha=0.8, linewidth=2)
        
        corr_coef = self.processed_data['discharge_upstream'].corr(self.processed_data['discharge_downstream'])
        plt.text(0.05, 0.95, f'Pearson r = {corr_coef:.4f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        plt.savefig('analysis_results/可视化_上下游径流散点图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 季节性箱线图 - 按春夏秋冬顺序
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 定义季节顺序
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        
        for i, col in enumerate(numeric_cols):
            # 创建DataFrame并排序
            plot_data = pd.DataFrame({
                'value': self.processed_data[col],
                'season': self.processed_data['season']
            })
            
            # 使用pandas的boxplot，但指定顺序
            box_data = []
            for season in season_order:
                box_data.append(plot_data[plot_data['season'] == season]['value'].dropna())
            
            # 绘制箱线图
            bp = axes[i].boxplot(box_data, patch_artist=True)
            
            # 设置颜色
            colors = ['#FFCCCC', '#CCFFCC', '#CCCCFF', '#FFFFCC']  # 浅红、浅绿、浅蓝、浅黄
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # 设置x轴标签
            axes[i].set_xticks(range(1, len(season_order) + 1))
            axes[i].set_xticklabels(season_order)
            
            axes[i].set_title(f'{col} 季节性变化', fontsize=10)
            axes[i].set_xlabel('Season')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3, linestyle='--')
        
        axes[-1].axis('off')
        fig.suptitle('Seasonal Variation Boxplots', fontsize=16)
        plt.tight_layout()
        plt.savefig('analysis_results/可视化_季节性箱线图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self):
        # 执行分析流程
        print("="*50)
        print("1. 基础信息分析")
        print("="*50)
        missing_info = self.basic_info()
        
        print("\n" + "="*50)
        print("2. 数据预处理")
        print("="*50)
        processed_data, outlier_info = self.data_preprocessing()
        
        print("\n" + "="*50)
        print("3. 描述性统计与分布检验")
        print("="*50)
        desc_stats, dist_tests = self.descriptive_statistics()
        
        print("\n" + "="*50)
        print("4. 相关性分析")
        print("="*50)
        corr_results = self.correlation_analysis()
        
        print("\n" + "="*50)
        print("5. 季节性分析")
        print("="*50)
        seasonal_results = self.seasonal_analysis()
        
        print("\n" + "="*50)
        print("6. 极值分析")
        print("="*50)
        extreme_results = self.extreme_value_analysis()
        
        print("\n" + "="*50)
        print("7. 生成可视化图表")
        print("="*50)
        self.generate_visualizations()
        
        # 分布特征总结
        upstream_skew = self.summary_stats['descriptive'].loc['skewness', 'discharge_upstream']
        downstream_skew = self.summary_stats['descriptive'].loc['skewness', 'discharge_downstream']
        print("\n" + "="*50)
        print("8. 分析总结")
        print("="*50)
        print("\n分布特征:")
        print(f"   - 上游径流偏度: {upstream_skew:.3f} ({'右偏' if upstream_skew > 0.5 else '近似对称' if abs(upstream_skew) < 0.5 else '左偏'})")
        print(f"   - 下游径流偏度: {downstream_skew:.3f} ({'右偏' if downstream_skew > 0.5 else '近似对称' if abs(downstream_skew) < 0.5 else '左偏'})")
        print(f"   - 重尾特征: {'明显' if max(abs(upstream_skew), abs(downstream_skew)) > 1 else '轻微'}")
        
        # 相关性分析总结
        upstream_downstream_corr = self.summary_stats['correlations']['upstream_downstream']
        print("\n变量关系:")
        print(f"   - 上下游径流线性相关: {upstream_downstream_corr['Pearson']:.4f}")
        print(f"   - 上下游径流秩相关: {upstream_downstream_corr['Spearman']:.4f}")
        print(f"   - 相关性强度: {'强' if upstream_downstream_corr['Pearson'] > 0.8 else '中等' if upstream_downstream_corr['Pearson'] > 0.5 else '弱'}")
        
        # 保存分析总结
        summary_dict = {
            '上游径流偏度': [round(upstream_skew, 3)],
            '上游径流偏度描述': ['右偏' if upstream_skew > 0.5 else '近似对称' if abs(upstream_skew) < 0.5 else '左偏'],
            '下游径流偏度': [round(downstream_skew, 3)],
            '下游径流偏度描述': ['右偏' if downstream_skew > 0.5 else '近似对称' if abs(downstream_skew) < 0.5 else '左偏'],
            '重尾特征': ['明显' if max(abs(upstream_skew), abs(downstream_skew)) > 1 else '轻微'],
            '上下游Pearson相关系数': [round(upstream_downstream_corr['Pearson'], 4)],
            '上下游Spearman相关系数': [round(upstream_downstream_corr['Spearman'], 4)],
            '相关性强度': ['强' if upstream_downstream_corr['Pearson'] > 0.8 else '中等' if upstream_downstream_corr['Pearson'] > 0.5 else '弱']
        }
        summary_df = pd.DataFrame(summary_dict)
        summary_df.to_csv('analysis_results/07_分析总结.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 所有分析结果已保存至 'analysis_results' 文件夹！")
        
        return {
            'processed_data': self.processed_data,
            'summary_stats': self.summary_stats,
            'recommendations': {
                'alpha_stable': True,
                'nonlinear_methods': ['SVR', 'RandomForest', 'PolynomialRegression'],
                'copula_candidates': ['Clayton', 'Gumbel', 't-Copula', 'Frank'],
                'seasonal_adjustment': True
            }
        }

# 运行分析
if __name__ == "__main__":
    # 修改为你的数据文件路径
    data_path = r"D:\降水与径流量\data\data1999-2024.csv"
    
    # 初始化分析器并运行
    analyzer = ColumbiaDataAnalyzer(data_path)
    results = analyzer.generate_report()