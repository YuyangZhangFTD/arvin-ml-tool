import json
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from arvin_python.causal_tool.metric.qini import cal_qini_score
from arvin_python.tf_nn_utils.keras_function_api_utils import *

eval_flag = 0
model_name = "slearner_nn"

# 数据集相关
IS_TREAT_COL = "treatment"  # treatment
FEATURE_COLS = [f"f{i}" for i in range(12)]  # feature
IS_EXPOSURE_COL = "exposure"  # whether treatment is exposure
IS_VISIT_COL = "visit"  # if visit=0, conversion=0
LABEL_COL = "conversion"  # label
TOTAL_COLS = FEATURE_COLS + [IS_TREAT_COL, LABEL_COL, IS_VISIT_COL, IS_EXPOSURE_COL]

# 超参数
VALID_SET_RATIO = 0.1
EPOCH_NUM = 5
BATCH_SIZE = 128
HIDDEN_LAYER_NUM = [16, 4]
LEARNING_RATE = 0.01

# 读数据
data_train = pd.read_csv(f"./criteo-uplift/tr_{eval_flag}.gz")
data_train, data_valid = train_test_split(data_train, test_size=VALID_SET_RATIO)
data_test = pd.read_csv(f"./criteo-uplift/te_{eval_flag}.gz")


# 构建data set
def make_ds_from_df(
        _df_data, feature_cols=None, label_col=None, batch_size=512, shuffle_buffer_size=10000,
        use_shuffle=True
):
    assert not (feature_cols is None or label_col is None)
    _ds = tf.data.Dataset.from_tensor_slices((
        _df_data[feature_cols].to_dict("list"),
        _df_data[[label_col]].to_dict("list")
    ))
    if use_shuffle:
        return _ds.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size=batch_size)
    else:
        return _ds.batch(batch_size=batch_size)


# def make_ds_from_file(
#         _file_path, feature_cols=None, label_col=None, batch_size=512, shuffle_buffer_size=10000,
#         use_shuffle=True
# ):
#     assert not (feature_cols is None or label_col is None)
#     return tf.data.experimental.make_csv_dataset(
#         file_pattern=[_file_path], select_columns=FEATURE_COLS + [IS_TREAT_COL, LABEL_COL],
#         label_name=LABEL_COL, batch_size=batch_size, compression_type="GZIP",
#         num_epochs=1  # 在外层控制epoch num，experimental中会默认无限重复
#     )


with open("tf_nn_conf/slearner_nn/feature_conf_raw.json") as f:
    feature_conf = json.load(f)["features"]

with open("tf_nn_conf/slearner_nn/model_conf.json") as f:
    model_conf = json.load(f)

feature_columns = generate_feature_columns(feature_conf)
feature_inputs = generate_feature_inputs(feature_conf)
block_layer_output = generate_feature_block_layer_output(
    model_conf["feature"], feature_columns=feature_columns, feature_inputs=feature_inputs
)

# y = main_net_output + main_net_tau * trt = f(f0,...f11) + g(f0,...,f11) * trt
tenser_in_main_net = block_layer_output["main_net"]
tensor_in_is_trt = block_layer_output["is_trt"]

for i, num in enumerate(HIDDEN_LAYER_NUM):
    tenser_in_main_net = keras.layers.Dense(
        units=num,
        # activation=keras.activations.relu,
        # kernel_regularizer=keras.regularizers.l2(0.01),
        # bias_regularizer=keras.regularizers.l2(0.01),
        name=f"hidden_{i}"
    )(tenser_in_main_net)
    tenser_in_main_net = keras.layers.LeakyReLU(alpha=0.1)(tenser_in_main_net)  # 如果想使用leaky_relu，需要当成新layer使用
    tenser_in_main_net = keras.layers.Dropout(rate=0.2)(tenser_in_main_net)
main_net_output = keras.layers.Dense(1, name="main_net_output")(tenser_in_main_net)
main_net_tau = keras.layers.Dense(1, name="main_net_tau")(tenser_in_main_net)
# keras.layers.Lambda(lambda x: x[0] + x[1] * x[2])([main_net_output, main_net_tau, tensor_in_is_trt])
logit = main_net_output + main_net_tau * tensor_in_is_trt
# keras.Model的outputs中的名字需要跟y中的名字一致
prob = keras.layers.Lambda(function=tf.keras.activations.sigmoid, name=LABEL_COL)(logit)
model = keras.Model(inputs=[v for v in feature_inputs.values()], outputs=prob)

model.compile(
    optimizer=tf.optimizers.Adagrad(learning_rate=LEARNING_RATE),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.AUC()],
)
print(model.summary())
# todo: tf 2.6 会报错，不知道为啥
# tf.keras.utils.plot_model(
#     model, to_file=f"111.png",
#     show_shapes=True,
#     show_layer_names=True
# )

# 开始训练
batch_num_valid = int(1 * 1e6 // BATCH_SIZE)    # 一百万样本 做一次 评估
ds_train = make_ds_from_df(
    data_train, feature_cols=FEATURE_COLS + [IS_TREAT_COL], label_col=LABEL_COL, batch_size=BATCH_SIZE
)
ds_valid = make_ds_from_df(
    data_valid, feature_cols=FEATURE_COLS + [IS_TREAT_COL], label_col=LABEL_COL, batch_size=int(1e5),
    use_shuffle=False
)
for epoch in range(EPOCH_NUM):

    print("=" * 10, "epoch: ", epoch, "=" * 10)

    for i, (X, y) in enumerate(ds_train):

        train_result = model.train_on_batch(X, y, reset_metrics=False)
        if i % 10 == 0: print("train: ", i, "train:", dict(zip(model.metrics_names, train_result)))

        if i % batch_num_valid == 0 and i > 0:
            valid_list = []
            for j, (X, y) in enumerate(ds_valid):

                valid_result = model.test_on_batch(X, y, reset_metrics=False)
                if j % 50 == 0: print("valid: ", j, "valid:", dict(zip(model.metrics_names, valid_result)))

                y_pred = model.predict_on_batch(X)
                valid_list.append((y[LABEL_COL].numpy(), y_pred))

            y_true = np.concatenate([_y[0] for _y in valid_list])
            y_pred = np.concatenate([_y[1] for _y in valid_list])
            print("sklearn auc: ", roc_auc_score(y_true, y_pred))

# 产出预测值 + 评估
plt.Figure()
plt.plot([0, 1], [0, 1], label='rand')
result_str = []
for _df, _name in zip([data_train, data_valid, data_test], ["train", "valid", "test"]):

    for is_trt, col in zip([0, 1], ["_ctl_prediction", "_trt_prediction"]):
        _df_copy = _df.copy()
        _df_copy["treatment"] = is_trt
        _ds_copy = make_ds_from_df(
            _df_copy, feature_cols=FEATURE_COLS + [IS_TREAT_COL], label_col=LABEL_COL,
            batch_size=BATCH_SIZE * 5, use_shuffle=False
        )
        _y_pred = list()
        for X, _ in _ds_copy:
            _y_pred.append(model.predict_on_batch(X))
        _df[col] = np.concatenate(_y_pred)

    _df["_is_trt"] = 1
    _df["_is_ctl"] = 0
    _df["_ite"] = _df["_trt_prediction"] - _df["_ctl_prediction"]

    _df, qini_score = cal_qini_score(_df, score_col="_ite", treatment_col=IS_TREAT_COL, outcome_col=LABEL_COL)

    plt.Figure()
    plt.plot(_df["percentile"], _df["normal_total_lift"], label=f"{model_name}")
    result_str.append(f"{_name}={round(qini_score, 4)}")

plt.legend()
plt.xlabel('percentile')
plt.ylabel(f'normalized uplift ({LABEL_COL})')
plt.title(f"{model_name}: {','.join(result_str)}")
plt.savefig(f"{model_name}.png")
plt.show()
# data_test.to_csv(f"./criteo-uplift/pred_{eval_flag}_{model_name}.gz", index=None, compression="gzip")
