import pandas as pd

def cal_qini_score(
    _df:pd.DataFrame, score_col='score', 
    treatment_col='treatment', outcome_col='conversion'
):
    _df = _df.copy()
    _df.sort_values(by=score_col, inplace=True, ascending=False)
    _df.loc[:, '_is_trt'] = _df[treatment_col]
    _df.loc[:, '_is_ctl'] = 1 - _df[treatment_col]
    _df.loc[:, '_trt_cum_y'] = (_df['_is_trt'] * _df[outcome_col]).cumsum()
    _df.loc[:, '_ctl_cum_y'] = (_df['_is_ctl'] * _df[outcome_col]).cumsum()
    _df.loc[:, '_trt_cum_cnt'] = _df['_is_trt'].cumsum()
    _df.loc[:, '_ctl_cum_cnt'] = _df['_is_ctl'].cumsum()
    _df.loc[:, '_total_cum_cnt'] = _df['_trt_cum_cnt'] + _df['_ctl_cum_cnt']
    _df.loc[:, 'total_lift'] = (
        (_df['_trt_cum_y'] / _df['_trt_cum_cnt'] - _df['_ctl_cum_y'] / _df['_ctl_cum_cnt']) * _df['_total_cum_cnt']
    ).fillna(0)
    _df.loc[:, 'normal_total_lift'] = _df['total_lift'] / _df['total_lift'].max()
    _df.loc[:, 'percentile'] = _df['_total_cum_cnt'] / _df.shape[0]
    
    qini_score = _df['normal_total_lift'].mean() - 0.5
    
    return _df, qini_score
