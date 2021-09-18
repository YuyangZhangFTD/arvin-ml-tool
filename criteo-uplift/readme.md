# Criteo-uplift 数据集实验

## 官方地址

[Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)

## 数据EDA

基础信息
- f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11: feature values (dense, float)
- treatment: treatment group (1 = treated, 0 = control)
- conversion: whether a conversion occured for this user (binary, label)
- visit: whether a visit occured for this user (binary, label)
- exposure: treatment effect, whether the user has been effectively exposed (binary)
- Format: CSV
- Size: 297M (compressed)
- Rows: 13,979,592
- Average Visit Rate: .046992
- Average Conversion Rate: .00292
- Treatment Ratio: .85

