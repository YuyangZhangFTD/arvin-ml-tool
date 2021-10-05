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
model_name = "nn_with_r_loss_reg"

# 数据集相关
IS_TREAT_COL = "treatment"  # treatment
FEATURE_COLS = [f"f{i}" for i in range(12)]  # feature
IS_EXPOSURE_COL = "exposure"  # whether treatment is exposure
IS_VISIT_COL = "visit"  # if visit=0, conversion=0
LABEL_COL = "conversion"  # label
TOTAL_COLS = FEATURE_COLS + [IS_TREAT_COL, LABEL_COL, IS_VISIT_COL, IS_EXPOSURE_COL]

# 超参数
VALID_SET_RATIO = 0.1
EPOCH_NUM = 2
BATCH_SIZE = 256
HIDDEN_LAYER_NUM = [16, 4]
LEARNING_RATE = 0.01
PROPENSITY_SCORE = 0.85 / 0.15

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


with open(f"tf_nn_conf/{model_name}/feature_conf_raw.json") as f:
    feature_conf = json.load(f)["features"]

with open(f"tf_nn_conf/{model_name}/model_conf.json") as f:
    model_conf = json.load(f)

feature_columns = generate_feature_columns(feature_conf)
feature_inputs = generate_feature_inputs(feature_conf)
block_layer_output = generate_feature_block_layer_output(
    model_conf["feature"], feature_columns=feature_columns, feature_inputs=feature_inputs
)

# y = main_net_output + main_net_tau * trt = f(f0,...f11) + g(f0,...,f11) * trt
tenser_main_net = block_layer_output["main_net"]
tensor_is_trt = block_layer_output["is_trt"]

for i, num in enumerate(HIDDEN_LAYER_NUM):
    tenser_main_net = keras.layers.Dense(
        units=num,
        # activation=keras.activations.relu,
        # kernel_regularizer=keras.regularizers.l2(0.01),
        # bias_regularizer=keras.regularizers.l2(0.01),
        name=f"hidden_{i}"
    )(tenser_main_net)
    tenser_main_net = keras.layers.LeakyReLU(alpha=0.1)(tenser_main_net)  # 如果想使用leaky_relu，需要当成新layer使用
    tenser_main_net = keras.layers.Dropout(rate=0.2)(tenser_main_net)

output_trt = keras.layers.Dense(1, name="trt_output")(tenser_main_net)
output_ctl = keras.layers.Dense(1, name="ctl_output")(tenser_main_net)

output_logit = output_trt * tensor_is_trt + output_ctl * (1 - tensor_is_trt)
# main_net_output = keras.layers.Dense(1, name="main_net_output")(tenser_main_net)
# main_net_tau = keras.layers.Dense(1, name="main_net_tau")(tenser_main_net)
# # keras.layers.Lambda(lambda x: x[0] + x[1] * x[2])([main_net_output, main_net_tau, tensor_in_is_trt])
# logit = main_net_output + main_net_tau * tensor_is_trt
# keras.Model的outputs中的名字需要跟y中的名字一致
prob = keras.layers.Lambda(function=tf.keras.activations.sigmoid, name=LABEL_COL)(output_logit)
model = keras.Model(inputs=[v for v in feature_inputs.values()], outputs=[prob, tensor_is_trt])

print(model.summary())
# todo: tf 2.6 会报错，不知道为啥
# tf.keras.utils.plot_model(
#     model, to_file=f"111.png",
#     show_shapes=True,
#     show_layer_names=True
# )


def loss_with_r_loss_reg(_y_true, _y_pred, _is_trt, from_logits=False, r_reg_coeff=1.0):
    if from_logits:
        _y_pred = tf.sigmoid(_y_pred)
    batch_cross_entropy = keras.losses.binary_crossentropy(_y_true, _y_pred, from_logits=False)
    ce_loss = tf.reduce_mean(batch_cross_entropy)
    r_reg = tf.reduce_mean((_is_trt / 0.85 - (1 - _is_trt) / 0.15) * _y_pred)
    detail = dict()
    detail['ce_loss'] = ce_loss
    detail['r_loss'] = r_reg
    return ce_loss + r_reg * r_reg_coeff, detail


optimizer = keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)

# 定义监控指标
auc_train = keras.metrics.AUC(name="auc_train")
loss_train = keras.metrics.Mean(name="loss_train")

auc_valid = keras.metrics.AUC(name="auc_valid")
loss_valid = keras.metrics.Mean(name="loss_valid")


@tf.function        # tf.function，构建图，加速运行
def train_step(_model, _features, _labels):
    with tf.GradientTape() as tape:
        _prob, _is_trt = _model(_features, training=True)
        loss, detail = loss_with_r_loss_reg(
            _labels, _prob, _is_trt,
            r_reg_coeff=model_conf['model']['r_loss_regularization_coefficient']
        )

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_train.update_state(loss)
    auc_train.update_state(_labels, _prob)
    return _prob, _is_trt, detail


@tf.function
def valid_step(_model, _features, _labels):
    _prob, _is_trt = _model(_features)
    loss, detail = loss_with_r_loss_reg(
        _labels, _prob, _is_trt,
        r_reg_coeff=model_conf['model']['r_loss_regularization_coefficient']
    )
    loss_valid.update_state(loss)
    auc_valid.update_state(_labels, _prob)
    return _prob, _is_trt, detail


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

        _ = train_step(model, X, tf.expand_dims(y["conversion"], -1))
        # train_result = model.train_on_batch(X, y, reset_metrics=False)
        if i % 100 == 0:
            print("train:", i,
                  "loss:", loss_train.result().numpy(),
                  "auc:", auc_train.result().numpy())

        if i % batch_num_valid == 0 and i > 0:
            valid_list = []
            for j, (X, y) in enumerate(ds_valid):

                y_pred, *_ = valid_step(model, X, tf.expand_dims(y["conversion"], -1))
                if j % 10 == 0:
                    print("valid: ", j,
                          "loss:", loss_valid.result().numpy(),
                          "auc:", auc_valid.result().numpy())

                valid_list.append((y[LABEL_COL].numpy(), y_pred.numpy()))

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
        _y_pred_list = list()
        for X, _ in _ds_copy:
            _y_pred_batch, _ = model.predict_on_batch(X)
            _y_pred_list.append(_y_pred_batch)
        _df[col] = np.concatenate(_y_pred_list)

    _df["_is_trt"] = 1
    _df["_is_ctl"] = 0
    _df["_ite"] = _df["_trt_prediction"] - _df["_ctl_prediction"]

    _df, qini_score = cal_qini_score(_df, score_col="_ite", treatment_col=IS_TREAT_COL, outcome_col=LABEL_COL)

    plt.Figure()
    plt.plot(_df["percentile"], _df["normal_total_lift"], label=f"{_name}")
    result_str.append(f"{_name}={round(qini_score, 4)}")

plt.legend()
plt.xlabel('percentile')
plt.ylabel(f'normalized uplift ({LABEL_COL})')
plt.title(f"{model_name}: {','.join(result_str)}")
plt.savefig(f"qini_curve_{model_name}.png")
# plt.show()
# data_test.to_csv(f"./criteo-uplift/pred_{eval_flag}_{model_name}.gz", index=None, compression="gzip")
