import tensorflow as tf
import tensorflow.keras as keras


def _parse_raw_data_type(rdt):
    if rdt == "float":
        return tf.float32
    elif rdt == "int":
        return tf.int32
    elif rdt == "string":
        return tf.string
    else:
        raise Exception(f"raw_data_type {rdt} is not implemented")


def _parse_normalize_fn(nf, param):
    if nf == "z_score":
        def _z_score(x):
            return (x - param["mean"]) / param["std"]

        return _z_score
    elif nf == "square":
        def _square(x):
            return x ** 2

        return _square
    else:
        def _fn(x):
            return x

        return _fn  # 直接 return None 会报错


def generate_feature_columns(feature_conf: dict = None):
    feature_columns = {}
    for _feat_conf in feature_conf:
        if _feat_conf["feature_type"] == "embedding" \
                and "input_size" in _feat_conf and _feat_conf["input_size"] == 1 \
                and "bucket_boundaries" in _feat_conf \
                and "embedding_size" in _feat_conf:
            feature_columns[_feat_conf["name"]] = tf.feature_column.embedding_column(
                categorical_column=tf.feature_column.bucketized_column(
                    source_column=tf.feature_column.numeric_column(_feat_conf["name"]),
                    boundaries=_feat_conf["bucket_boundaries"]
                ), dimension=_feat_conf["embedding_size"]
            )
        elif _feat_conf["feature_type"] == "embedding" \
                and "input_size" in _feat_conf and _feat_conf["input_size"] == 1 \
                and "bucket_boundaries" not in _feat_conf \
                and "hash_bucket_size" in _feat_conf \
                and "embedding_size" in _feat_conf:
            feature_columns[_feat_conf["name"]] = tf.feature_column.embedding_column(
                categorical_column=tf.feature_column.categorical_column_with_hash_bucket(
                    key=_feat_conf["name"], hash_bucket_size=_feat_conf["hash_bucket_size"]
                ), dimension=_feat_conf["embedding_size"]
            )
        elif _feat_conf["feature_type"] == "embedding" \
                and "input_size" in _feat_conf and _feat_conf["input_size"] == 1 \
                and "bucket_boundaries" not in _feat_conf \
                and "hash_bucket_size" not in _feat_conf \
                and "max_index_num" in _feat_conf \
                and "embedding_size" in _feat_conf:
            feature_columns[_feat_conf["name"]] = tf.feature_column.embedding_column(
                categorical_column=tf.feature_column.categorical_column_with_identity(
                    key=_feat_conf["name"], num_buckets=_feat_conf["max_index_num"]
                ), dimension=_feat_conf["embedding_size"]
            )
        elif _feat_conf["feature_type"] == "raw":
            feature_columns[_feat_conf["name"]] = tf.feature_column.numeric_column(
                _feat_conf["name"], shape=(_feat_conf["input_size"],),
                normalizer_fn=_parse_normalize_fn(
                    _feat_conf.get("normalize_fn", None), _feat_conf.get("normalize_fn_param", None)
                )
            )
    return feature_columns


def generate_feature_inputs(feature_conf: dict = None):
    feature_inputs = {}
    for _feat_conf in feature_conf:
        feature_inputs[_feat_conf["name"]] = keras.layers.Input(
            (_feat_conf["input_size"],),
            name=_feat_conf["name"],
            dtype=_parse_raw_data_type(_feat_conf["raw_data_type"])
        )
    return feature_inputs


def generate_feature_block_layer_output(block_conf: dict = None, feature_columns: dict = None,
                                        feature_inputs: dict = None):
    block_layer = dict()
    for _block_name, _block_feature_list in block_conf.items():
        _block_feature_columns = [feature_columns[_name] for _name in _block_feature_list]
        _block_feature_inputs = {k: v for k, v in feature_inputs.items() if k in _block_feature_list}
        block_layer[_block_name] = keras.layers.DenseFeatures(
            _block_feature_columns, name=f"{_block_name}_dense_feature"
        )(_block_feature_inputs)
    return block_layer
