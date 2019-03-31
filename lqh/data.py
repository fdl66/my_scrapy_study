#!coding=utf8
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

# 对变量区分类型,获取数值变量、分类变量、text 变量列表
type_df=pd.read_excel(data_src+'数据类型描述.xlsx')
eid = 'eventid'

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
    # res_df.to_excel(data_des+'data%s.xlsx'%data_type,index=False)
    return res_df


def get_data_stat_info(df, note='name'):
    rows = df.shape[0]
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
    # 删除缺失比例超过 85%的列（变量）
    df1 = del_over_miss_cols(df, thresh=0.85)
    #  删除缺失比例超过 50%的行（样本）
    res_df = del_over_miss_rows(df1, thresh=0.5)
    print('清洗后:', res_df.shape)
    # res_df.to_excel(data_des+'data_cleaned.xlsx',index=False)
    return res_df


# C 题第一问的建模数据
def get_data_Q1(df):
    df = df.copy()
    feats_df = pd.read_excel(data_src + 'q1_变量纳排标准.xlsx')
    feats = feats_df[feats_df['是否使用'] == '√']['变量（英文名）']
    print('q1_变量纳排标准:', feats)
    feats = [col for col in feats if col in list(df.columns)]
    print('第一问使用的变量：', feats)
    res_df = df[feats]
    print(res_df.shape)
    # 对数据进行处理
    data_Q1_process(res_df)
    return res_df


def data_Q1_process(df):
    # df.drop(['eventid'],axis=1,inplace=True)
    df = del_over_miss_rows(df, thresh=0.5)
    numer_feats = [col for col in df.columns if col in numer_vars]
    need_dummy_feats = [col for col in df.columns if col in need_dummy_vars]
    for col in need_dummy_feats:
        df[col] = df[col].apply(lambda x: -1 if x in [-9, -99, -999] else x)

    scaler = StandardScaler()
    # 空缺值的填补
    df.fillna(-1, inplace=True)
    # 连续数值进行标准化
    df[numer_feats] = scaler.fit_transform(df[numer_feats])

    # 分类变量进行哑编码
    df = pd.get_dummies(df, columns=need_dummy_feats)
    df.to_excel(data_src + 'data_forQ1_标准化后.xlsx', index=False)
    return df


# 连续变量之间计算 Pearson 相关系数
def calc_numer_corr(df):
    numer_cols = [col for col in numer_vars if col in list(df.columns)]
    corr_df = df[numer_cols].corr(method='pearson')
    corr_df.to_excel(data_des + '数值变量相关系数.xlsx', index=True)


# 分类变量使用卡方检验，检验 p 值是否显著
def chi_square_test(df0, target):
    df = df0.fillna(-999)
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
        stat, pvalue, dof, expected = chi2_contingency(cross_table, correction=False)
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
    # 我不知道^是什么意思
    if expected.min() < 1 or len([v for v in expected.reshape(1, -1)[0] if v < 5]) /\
            (expected.shape[0] * expected.shape[1]) > 0.2:
        iswarning = 'warning'
        stat = round(stat, 3)
        pvalue = round(pvalue, 3)
    if pvalue == 0:
        pvalue = '<0.001'
    return stat, pvalue, iswarning


# 从原始的 11W 行数据中筛选出 Q1_test_sample 中的 9 个样本
def get_q1_9sample(df):
    q1_test = pd.read_excel(data_src + 'q1_test_sample.xlsx')
    res_df = df[df['eventid'].isin(list(q1_test['eventid']))]
    res_df.to_excel(data_src + 'q1_9sample.xlsx', index=False)
    print('q1_9sample 提取完毕')


# 从原始的 11W 行数据中筛选出 Q2_test_sample 中的 9 个样本
def get_q2_10sample(df):
    q2_test = pd.read_excel(data_src + 'q2_test_sample.xlsx')
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


if __name__ == "__main__":
    import pdb;pdb.set_trace()
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
    # 生成问题 2 的数据
    get_data_Q2(data)
    get_data_Q3(data)
    # get_q1_9sample(data)
    # get_q2_10sample(data)
