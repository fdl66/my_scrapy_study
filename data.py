import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, fisher_exact
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FactorAnalysis
import warnings
# import mpl_toolkits
warnings.filterwarnings('ignore')


data_src = '../data/'
data_des = '../temp/'

# 对变量区分类型,获取数值变量、分类变量、text 变量列表 type_df=pd.read_excel(data_src+'数据类型描述.xlsx') eid='eventid'
numer_vars = list(type_df[type_df['类型'] == '数值']['变量（英文名）'])
cls_vars = list(type_df[type_df['类型'] == '分类']['变量（英文名）'])
text_vars = list(type_df[type_df['类型'] == 'text']['变量（英文名）'])
bi_cls_vars = ['extended', 'crit1', 'crit2', 'crit3', 'doubtterr', 'multiple',
               'success', 'suicide', 'guncertain1', 'guncertain2', 'guncertain3', 'claim3', 'weaptype4']
need_dummy_vars = [col for col in cls_vars if col not in bi_cls_vars]


# 根据每个变量的不同的个数判断每个变量的类型：数值或者是分类
def data_type_identify(df):
    col_type_list = []
    col_uni_nums = []
    for c in data.columns:
        col_uni_num = len(data[c].value_counts())
        col_uni_nums.append(col_uni_num)
        col_type = None
        if col_uni_num > 50:
            col_type = '连续'
        else:
            col_type = '分类'
        col_type_list.append(col_type)

        type_df = pd.DataFrame()
        type_df['变量'] = list(data.columns)
        type_df['取值个数'] = col_uni_nums
        type_df['类型'] = col_type_list
        type_df.to_excel(data_des + 'meta.xlsx', index=False)
    return type_df


# 删除掉一些出现次数低，缺失比例大的字段，保留超过阈值的特征
def del_over_miss_cols(df, thresh=0.85):
    print('移除之前总的字段数量', df.shape[1])


    rows = df.shape[0]
    df_null = df.isnull()
    col_miss_ser = df_null.sum(axis=0)
    drop_col_list = [i for i in col_miss_ser.index if col_miss_ser[i] / rows > thresh]
    df.drop(labels=drop_col_list, axis=1, inplace=True)
    return df


# 删除缺失比例超过一定阈值的行，以及删除死亡总数缺失的行
def del_over_miss_rows(df, thresh=0.5):
    cols = df.shape[1]
    print('删除缺失比例超过%s 的行' % thresh)
    df_null = df.isnull()
    rows_miss_series = df_null.sum(axis=1)
    # print(rows_miss_series)
    drop_row_index = [i for i in rows_miss_series.index if rows_miss_series[i] / cols > thresh]
    df.drop(labels=drop_row_index, axis=0, inplace=True)
    # 删除死亡总数缺失的行
    df['nkill'] = df['nkill'].fillna(-999)
    nkill_miss_index = df[df['nkill'] == -999].index.tolist()
    df.drop(labels=nkill_miss_index, axis=0, inplace=True)
    print('删除死亡总数缺失的行数：%s' % len(nkill_miss_index))
    return df


def get_data(df, data_type='数值变量'):
    col_df = pd.read_excel(data_src + 'data_col.xlsx')
    if data_type == '数值变量':
        cols = list(col_df['数值变量'])[0:24]
    elif data_type == 'text 变量':
        cols = list(col_df['text 变量'])[:28]
    elif data_type == '分类变量':
        cols = list(col_df['分类变量'])[:54]
    cols_use = [col for col in cols if col in df.columns]
    res_df = df[cols_use]


# res_df.to_excel(data_des+'data%s.xlsx'%data_type,index=False) return res_df

def get_data_stat_info(df, note='name'): rows = df.shape[0]


stat_df0 = pd.read_excel(data_src + '数据类型统计.xlsx')
stat_df = stat_df0.copy()
cols = list(stat_df['变量（英文名）'])
cols = [col for col in cols if col in list(df.columns)]
stat_df = stat_df[stat_df['变量（英文名）'].isin(cols)]
miss_num_list = []
miss_ratio_list = []
uniq_vals_list = []
for col in cols:
    print(col)
    miss_num = df[col].isnull().sum()

miss_ratio = miss_num / rows * 100
uniq_vals = len(df[col].value_counts())
miss_num_list.append(miss_num)
miss_ratio_list.append(str(round(miss_ratio, 3)) + '%')
uniq_vals_list.append(uniq_vals)
stat_df['缺失数量'] = miss_num_list
stat_df['缺失比例'] = miss_ratio_list
stat_df['唯一取值个数'] = uniq_vals_list
stat_df.to_excel(data_des + '数据统计结果_%s.xlsx' % note, index=False)


def get_data_cols(df):
    res_df = pd.DataFrame()
    res_df['变量'] = list(df.columns)
    res_df.to_excel(data_des + 'data_cols_list.xlsx', index=False)


def data_clean(df):

# 删除缺失比例超过 85%的列（变量） df1=del_over_miss_cols(df,thresh=0.85)
#  删除缺失比例超过 50%的行（样本）
    res_df = del_over_miss_rows(df1, thresh=0.5)
    print('清洗后:', res_df.shape)
# res_df.to_excel(data_des+'data_cleaned.xlsx',index=False) return res_df

# C 题第一问的建模数据
def get_data_Q1(df): df = df.copy()


feats_df = pd.read_excel(data_src + 'q1_变量纳排标准.xlsx')
feats = feats_df[feats_df['是否使用'] == '√']['变量（英文名）']
print('q1_变量纳排标准:', feats)
feats = [col for col in feats if col in list(df.columns)]
print('第一问使用的变量：', feats)
res_df = df[feats]
print(res_df.shape)


# 对数据进行处理 data_Q1_process(res_df) return res_df

def data_Q1_process(df):


# df.drop(['eventid'],axis=1,inplace=True) df=del_over_miss_rows(df,thresh=0.5)
    numer_feats = [col for col in df.columns if col in numer_vars]
    need_dummy_feats = [col for col in df.columns if col in need_dummy_vars]
    for col in need_dummy_feats:

        df[col] = df[col].apply(lambda x: -1 if x in [-9, -99, -999] else x)

        scaler = StandardScaler()
        # 空缺值的填补
        df.fillna(-1, inplace=True)
        # 连续数值进行标准化
        df[numer_feats] = scaler.fit_transform(df[numer_feats])


# 分类变量进行哑编码 df=pd.get_dummies(df,columns=need_dummy_feats) df.to_excel(data_src+'data_forQ1_标准化后.xlsx',index=False) return df

# 连续变量之间计算 Pearson 相关系数
def calc_numer_corr(df):
    numer_cols = [col for col in numer_vars if col in list(df.columns)]
    corr_df = df[numer_cols].corr(method='pearson')
    corr_df.to_excel(data_des + '数值变量相关系数.xlsx', index=True)


# 分类变量使用卡方检验，检验 p 值是否显著
def chi_square_test(df0, target): df = df0.fillna(-999)


nan_value = -999
df = df[df[target] != nan_value]
target_size = len(df[target].value_counts())
vals_list = []
category_feats = [col for col in cls_vars if col in df.columns and col != target]
for col in category_feats:
    col_series = df[df[col] != nan_value][col]
    col_count = col_series.count()
    col_size = len(col_series.value_counts())
    data_kf = df[df[col] != nan_value][[col, target]]
cross_table = data_kf.groupby([col, target])[target].count().unstack()
cross_table.fillna(0, inplace=True)
if target_size == 2 and col_size == 2:
    stat, pvalue = foursquare_chi_test(cross_table, col_count)
else:
    stat, pvalue, iswarning = not_foursquare_chi_test(cross_table)
print('stat:', stat, 'pvalue:', pvalue)
vals_list.append([stat, pvalue])
chi_res = pd.DataFrame(data=vals_list, columns=['stat', 'pvalue'], index=category_feats)
chi_res.to_excel(data_des + '卡方检验结果.xlsx', index=True)
print('卡方检验结束')


def foursquare_chi_test(cross_table, col_count):
    stat, pvalue, dof, expected = chi2_contingency(cross_table, correction=False)
    if col_count >= 40 and expected.min() >= 5:
    # Pearson 卡方进行检验
        stat, pvalue, dof, expected = chi2_contingency(cross_table,correction=False)
    elif col_count >= 40 and expected.min() < 5 and expected.min() >= 1:
    # 连续性校正的卡方进行检验
        stat, pvalue, dof, expected = chi2_contingency(cross_table, correction=True)
    else:
    # 用 Fisher’s 检验
        stat, pvalue = fisher_exact(cross_table)

    stat = round(stat, 3)
    pvalue = round(pvalue, 3)
    if pvalue == 0:
        pvalue = '<0.001'
    return stat, pvalue


def not_foursquare_chi_test(cross_table):
    iswarning = ''
    stat, pvalue, dof, expected = chi2_contingency(cross_table, correction=False)
    #我不知道^是什么意思
    if expected.min() < 1 or len([v for v in expected.reshape(1, -1)[0] if v < 5])^(expected.shape[0] * expected.shape[1]) > 0.2:
        iswarning = 'warning'
        stat = round(stat, 3)
        pvalue = round(pvalue, 3)
    if pvalue == 0:
        pvalue = '<0.001'
    return stat, pvalue, iswarning


# 从原始的 11W 行数据中筛选出 Q1_test_sample 中的 9 个样本
def get_q1_9sample(df): q1_test = pd.read_excel(data_src + 'q1_test_sample.xlsx')


res_df = df[df['eventid'].isin(list(q1_test['eventid']))]
res_df.to_excel(data_src + 'q1_9sample.xlsx', index=False)
print('q1_9sample 提取完毕')


# 从原始的 11W 行数据中筛选出 Q2_test_sample 中的 9 个样本
def get_q2_10sample(df): q2_test = pd.read_excel(data_src + 'q2_test_sample.xlsx')


res_df = df[df['eventid'].isin(list(q2_test['eventid']))]
res_df.to_excel(data_src + 'q2_10sample.xlsx', index=False)
print('q2_10sample 提取完毕')


# 问题 2 中使用关于 2015、2016 年的数据
def get_data_Q2(df):


    data_q2 = df[df['iyear'].isin([2015, 2016])]
    print(data_q2.shape)
    data_q2.to_excel(data_src + 'data_q2 原数据.xlsx', index=False)
    get_data_stat_info(data_q2, 'Q2 原始')
    data_q2_cleaned = data_clean(data_q2)
    get_data_stat_info(data_q2_cleaned, 'Q2 删除行列后')
    data_q2_cleaned.to_excel(data_des + 'data_forQ2.xlsx', index=False)
    print('Q2 数据截取成功')


# 问题 3 中使用关于 2015、2016、2017 年的数据
def get_data_Q3(df):
    res_df = df[df['iyear'].isin([2015, 2016, 2017])]
    print(res_df.shape)
    res_df.to_excel(data_des + 'data_forQ3.xlsx', index=False)

if name == "  main  ":
    data_file = data_src + '附件 1.xlsx'
    data = pd.read_excel(data_file)
    # 生成数据统计表信息
    get_data_stat_info(data, '原始')
    data = data_clean(data)
    get_data_stat_info(data, '删除行列后')
    calc_numer_corr(data)
    chi_square_test(data, target='propextent')
    # 生成问题 1 的数据表
    get_data_Q1(data)
    # 生成问题 2 的数据 get_data_Q2(data) get_data_Q3(data)
    # get_q1_9sample(data)
    # get_q2_10sample(data)

#########################模型 1 代码##########################
# coding:utf-8 import numpy as np import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

data_src = '../data/'
data_des = '../temp/'

# 对数据做主成分分析 PCA
def pca_func(df):
    data_id = df['eventid']
    pca = PCA(n_components=0.8, svd_solver='full')
    X_df = df.drop(['eventid'], axis=1)
    print('放入 PCA 的数据：\n', X_df.head())
    df_matrix = pca.fit_transform(X_df.values)
    # 可解释变量的个数（主成分个数） pc_nums=len(pca.explained_variance_) cols=['主成分%s'%i for i in range(1,pc_nums+1)] with open(data_des+'pca_result.txt','w') as f:
    f.write('explained_variance_\n')
    f.write(str(pca.explained_variance_))
    f.write('\nexplained_variance_ratio_\n')
    f.write(str(pca.explained_variance_ratio_))
    f.write('\n 特征根\n')
    f.write(str(np.sqrt(pca.singular_values_)))
    f.close()

    covar_df = pd.DataFrame(data=pca.get_covariance(), columns=list(X_df.columns), index=list(X_df.columns))
    pca_df = pd.DataFrame(data=df_matrix, columns=cols, index=data_id.values)
    # 计算对各个组成分的值相加得到'F 值'(最终量化分级的标准) pca_df['F 值']=pca_df[cols].sum(axis=1) pca_df=pca_df.reset_index().rename(columns={'index':'eventid'})
    component_df = pd.DataFrame(data=pca.components_, columns=X_df.columns, index=cols)
    component_df['可解释比例'] = pca.explained_variance_ratio_
    component_df = component_df[['可解释比例'] + list(component_df.columns)]
    writer = pd.ExcelWriter(data_des + 'PCA_结果.xlsx')
    covar_df.to_excel(writer, sheet_name='协方差')
    pca_df.to_excel(writer, sheet_name='PCA_结果', index=False)
    component_df.to_excel(writer, sheet_name='主成分')
    writer.close()
    return pca_df


# 对数据样本进行聚类分成 5 类对应五个等级
def kmean_func(df, col='F 值'): kmean = KMeans(n_clusters=5)


cluster_res = kmean.fit_transform(df[col].values)
print('score:\n', kmean.score(df[col].values))
res_df = pd.DataFrame(data=cluster_res, index=df['eventid'],
columns=['分级 1', '分级 2', '分级 3','分级 4','分级 5'])
res_df = res_df.reset_index().rename(
    columns={'index': 'eventid'})
res_df['分级'] = kmean.labels_

res_df['分级'] = res_df['分级'].apply(lambda x: x + 1)
res_df = pd.merge(df, res_df, on=['eventid'])
#  重新按照'F 值'升序,对分级标签进行修正
# 对每个分级标签取出范围标签
range_list = []
for i in range(1, 6): left = res_df[res_df['分级'] == i]['F 值'].min()
right = res_df[res_df['分级'] == i]['F 值'].max()
range_list.append(tuple([left, right]))
range_list.sort()
res_df['分级'] = res_df['F 值'].apply(lambda x: rank_label(x, range_list))
res_df.to_excel(data_des + 'q1_聚类分级结果.xlsx', index=False)

# 对分级指标进行重新排列


def rank_label(x, range_list):
    for i in range(len(range_list)):
        if x >= range_list[i][0] and x <= range_list[i][1]: return 5 - i


if name == "  main  ": data_file = data_src + 'data_forQ1_标准化后.xlsx'
data = pd.read_excel(data_file)
print(data.head())
pca_df = pca_func(data)
kmean_func(pca_df)
#######################问题 2 模型代码###########################
# coding:utf-8 import numpy as np import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import FactorAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.cluster import DBSCAN
import warnings
import os
from data_process.data_preprocess import data_clean, del_over_miss_rows

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决中文字体负号显示不正常问题

data_src = '../data/'
data_des = '../temp/'

data_img = '../img/'

# 对变量区分类型,获取数值变量、分类变量、text 变量列表 type_df=pd.read_excel(data_src+'数据类型描述.xlsx') eid='eventid'
numer_vars = list(type_df[type_df['类型'] == '数值']['变量（英文名）'])
cls_vars = list(type_df[type_df['类型'] == '分类']['变量（英文名）'])
text_vars = list(type_df[type_df['类型'] == 'text']['变量（英文名）'])
bi_cls_vars = ['extended', 'crit1', 'crit2', 'crit3', 'doubtterr', 'multiple',
               'success', 'suicide', 'guncertain1', 'guncertain2', 'guncertain3', 'claim3', 'weaptype4']
need_dummy_vars = [col for col in cls_vars if col not in bi_cls_vars]


#  对 data 进行基础分析
def data_analysis(df): writer = pd.ExcelWriter(data_des + 'dataQ2_分析.xlsx')


# 对犯罪集团名字 gname 信息统计分析 gname_ser=df['gname'].value_counts().sort_values(ascending=False) gname_ser.to_excel(writer,sheet_name='gname_val_counts') writer.close()
# 对袭击事件声称负责 claimed 进行分析
# draw_pie(df,col='claimed') ser=df['claimed'].value_counts() plt.pie(ser.values) plt.savefig(data_img+'claimed.png')

# 分析分类变量的饼图比例
def draw_pie(df, col): df[col].value_counts().plot(kind='pie')


plt.title('%s' % col)
plt.savefig(data_img + '%s_ratio.png' % col)


#  对第 2 问拆分训练集和测试集
def get_data_Q2(df):
    df = df.copy()
    feats_df = pd.read_excel(data_src + 'q2_变量纳排标准.xlsx')
    feats = feats_df[feats_df['是否使用'] == '√']['变量（英文名）']
    print('q2_变量纳排标准:', feats)
    feats = [col for col in feats if col in df.columns]
    print('第 2 问使用的变量：', feats)
    res_df = df[feats]
    print(res_df.shape)
    # 对数据进行处理
    res_df2 = data_Q2_process(res_df)
    return res_df2


def data_Q2_process(df):
# df.drop(['eventid'],axis=1,inplace=True)
# df=del_over_miss_rows(df,thresh=0.5)
# 对 gname 进行处理，将次数小于 20 一下的单独分为一类"其他" gname_ser=df['gname'].value_counts().sort_values(ascending=False) df['gname']=df['gname'].apply(lambda x: 'other' if gname_ser[x]<20 else x) enc=LabelEncoder()
    df['gname'] = enc.fit_transform(df['gname'].values)

    numer_feats = [col for col in df.columns if col in numer_vars]
    numer_feats.remove('imonth')
    need_dummy_feats = [col for col in df.columns if col in need_dummy_vars]
    for col in need_dummy_feats:
        df[col] = df[col].apply(lambda x: -1 if x in [-9, -99, -999] else x)

        scaler = StandardScaler()
        # 空缺值的填补
        df.fillna(-1, inplace=True)
        # 连续数值进行标准化
        df[numer_feats] = scaler.fit_transform(df[numer_feats])
        # 分类变量进行哑编码 need_dummy_feats.append('gname') need_dummy_feats.append('imonth') df=pd.get_dummies(df,columns=need_dummy_feats)
        df.to_excel(data_src + 'data_forQ2_标准化后.xlsx', index=False)
        print(df.head())
        print(df.shape)
    return df

# 对数据做主成分分析 PCA
def factor_analysis(df0):
    df = df0.copy()
    data_id = df['eventid']
    n_component = 6
    fa = FactorAnalysis(n_components=n_component)
    X_df = df.drop(['eventid'], axis=1)
    print('放入因子分析的数据：\n', X_df.head())
    fa.fit(X_df.values)
    X_new = fa.transform(X_df.values)

    cols = ['因子%s' % i for i in range(1, n_component + 1)]

    covar_df = pd.DataFrame(data=fa.get_covariance(), columns=list(X_df.columns), index=list(X_df.columns))
    fa_res_df = pd.DataFrame(data=X_new, columns=cols, index=data_id.values)
    fa_res_df = fa_res_df.reset_index().rename(columns={'index': 'eventid'})
    component_df = pd.DataFrame(data=fa.components_, columns=X_df.columns, index=cols)
    writer = pd.ExcelWriter(data_des + 'FA_结果.xlsx')
    fa_res_df.to_excel(writer, sheet_name='FA_结果', index=False)
    component_df.to_excel(writer, sheet_name='因子')
    writer.close()
    return fa_res_df


# 对数据样本进行聚类分成 5 类对应五个等级
def dbscan_func(df0):
    df = df0.copy()
    data_id = df['eventid']
    X_df = df.drop(['eventid'], axis=1)
    dbscan = DBSCAN(min_samples=15, eps=0.8, leaf_size=30, n_jobs=-1)
    cluster_res = dbscan.fit_predict(X_df.values)
    res_df = pd.DataFrame(data=cluster_res, index=df['eventid'])
    res_df = res_df.reset_index().rename(columns={'index': 'eventid'})
    res_df.drop(labels=[0], axis=1, inplace=True)
    res_df['分级'] = dbscan.labels_
    res_df = pd.merge(df, res_df, on=['eventid'])
    # 把原数据中的 gname 值加上去 res_df=pd.merge(res_df,q2_src_tes[['eventid','gname']],on=['eventid']) res_df['分级']=res_df['分级'].apply(lambda x:x+1 if x!=-1 else x) res_df.to_excel(data_des+'q2_聚类结果.xlsx',index=False)
    return res_df


# 对聚类结果画图,柱状图和饼图
def draw_result(df): cluster_rank = df['分级'].value_counts().sort_values(ascending=False)


print(cluster_rank)
cluster_rank.plot(kind='bar')
plt.title('聚类结果占比')
plt.xlabel('簇的编号')
plt.ylabel('事件数量')
plt.xticks(ha='left', rotation=0)
plt.savefig(data_img + 'cluster_bar.png')
plt.close()

for index in cluster_rank.index.tolist()[0:5]: ser = df[df['分级'] == index]['gname'].value_counts().sort_values(
    ascending=False)[0:8]
print(ser)
ser.plot(kind='pie', title='第%s  个簇的饼图' % index, legend=True, use_index=False,
         figsize=(8, 8), x=300, y=500, rot=45)
plt.savefig(data_img + '%s.png' % index)
plt.close()
print('画图完毕！')
select_top_ter(df)
print('筛选恐怖组织完毕！')


#	筛选 top5 的恐怖组织或个人
def select_top_ter(df): cluster_rank = df['分级'].value_counts().sort_values(ascending=False)[0:5]


weight_sum = cluster_rank.sum()
cluster_weight = [v / weight_sum for v in cluster_rank]
ter_danger_dict = {}
for index in cluster_rank.index.tolist()[0:5]: ser = df[df['分级'] == index]['gname'].value_counts().sort_values(
    ascending=False)[0:8]
for j in ser.index:
    ser_sum = ser.sum()
    w = ser[j] / ser_sum
if j not in ter_danger_dict.keys():
    ter_danger_dict[j] = w
else:
    ter_danger_dict[j] += w
ter_danger_ser = pd.Series(ter_danger_dict, name='危害得分').sort_values(ascending=False)
ter_danger_ser.to_excel(data_des + '恐怖组织排序.xlsx', index=True)

if name == "     main    ":
    q2_src = pd.read_excel(data_src + 'data_q2 原数据.xlsx')
    q2_test_sample = pd.read_excel(data_src + 'q2_10sample.xlsx')
    q2_src_tes = pd.concat([q2_src, q2_test_sample], axis=0)

file_name = data_src + 'data_forQ2_标准化后.xlsx'
if os.path.exists(file_name):
    q2_df = pd.read_excel(file_name)

else:
    data = pd.read_excel(data_src + 'data_q2 原数据.xlsx')
    # data_analysis(data)
    # data=data_clean(data)
    data_q2 = data[data['gname'] != 'Unknown']
    # 把第二问中的 2017 年的 10 个样例进行合并

data_q2 = pd.concat([data_q2, q2_test_sample], axis=0)
q2_df = get_data_Q2(data_q2)
fa_df = factor_analysis(q2_df)
res_df = dbscan_func(fa_df)
draw_result(res_df)

