from arvin_python.causal_tool.metric.qini import cal_qini_score

import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

eval_flag = 0

# 超参数
EVAL_SET_RATIO = 0.2
LGB_MODEL_PARAMETER = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'binary',  # 目标函数
    'metric': {
        'auc',
        #         'binary_logloss'
    },  # 评估函数
    'num_iterations': 500,
    'max_depth': 4,
    'num_leaves': 64,  # 叶子节点数
    'learning_rate': 0.01,  # 学习速率
    #     'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.6,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging

    'early_stopping_rounds': 5,
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

# 数据集相关
IS_TREAT_COL = "treatment"  # treatment
FEATURE_COLS = [f"f{i}" for i in range(12)]  # feature
IS_EXPOSURE_COL = "exposure"  # whether treatment is exposure
IS_VISIT_COL = "visit"  # if visit=0, conversion=0
LABEL_COL = "conversion"  # label

# 读数据
data_train = pd.read_csv(f"./criteo-uplift/tr_{eval_flag}.gz")
data_train, data_eval = train_test_split(data_train, test_size=EVAL_SET_RATIO)
data_test = pd.read_csv(f"./criteo-uplift/te_{eval_flag}.gz")

# LGB 数据集转换
data_lgb_train = lgb.Dataset(data_train[FEATURE_COLS + [IS_TREAT_COL]], data_train[LABEL_COL])
data_lgb_eval = lgb.Dataset(data_eval[FEATURE_COLS + [IS_TREAT_COL]], data_eval[LABEL_COL])

# S-learner 训练
s_learner_lgb = lgb.train(
    params=LGB_MODEL_PARAMETER,
    train_set=data_lgb_train,
    valid_sets=[data_lgb_train, data_lgb_eval],
    valid_names=['train_set', 'eval_set'],
)

# 产出预测值 + 评估
for _df, _name in zip([data_train, data_eval, data_test], ['train', 'eval', 'test']):
    _df['_is_trt'] = 1
    _df['_is_ctl'] = 0
    _df['_trt_prediction'] = s_learner_lgb.predict(_df[FEATURE_COLS + ['_is_trt']])
    _df['_ctl_prediction'] = s_learner_lgb.predict(_df[FEATURE_COLS + ['_is_ctl']])
    _df['_ite'] = _df['_trt_prediction'] - _df['_ctl_prediction']

    _df, qini_score = cal_qini_score(_df, score_col='_ite', treatment_col=IS_TREAT_COL, outcome_col=LABEL_COL)

    plt.plot(_df['percentile'], _df['normal_total_lift'], label=f"s-learner")
    plt.plot([0, 1], [0, 1], label='rand')
    plt.legend()
    plt.xlabel('percentile')
    plt.ylabel(f'normalized uplift ({LABEL_COL})')
    plt.title(f's-learner_{_name}: {round(qini_score, 4)}')
    plt.show()

data_test.to_csv(f"./criteo-uplift/pred_{eval_flag}.gz", index=None, compression='gzip')
