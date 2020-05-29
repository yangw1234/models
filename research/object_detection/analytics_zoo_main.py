import tensorflow as tf
import sys

from object_detection import model_hparams
from object_detection import model_lib
from zoo.tfpark import TFEstimator, TFDataset
from zoo import init_nncontext
import os
sc = init_nncontext()

config = tf.estimator.RunConfig(model_dir="/tmp/object_detection")


dir_path = os.path.dirname(os.path.realpath(__file__))

pipeline_fname = os.path.join(dir_path, "samples/configs/ssd_mobilenet_v1_pets.config")

num_steps = 100

global_batch_size = 24


train_and_eval_dict = model_lib.create_estimator_and_inputs(
    run_config=config,
    hparams=model_hparams.create_hparams(None),
    pipeline_config_path=pipeline_fname,
    config_override="train_config {batch_size: 0}",
    train_steps=200,
    sample_1_of_n_eval_examples=5,
    sample_1_of_n_eval_on_train_examples=(
        5))

estimator = train_and_eval_dict['estimator']
train_input_fn = train_and_eval_dict['train_input_fn']
# eval_input_fns = train_and_eval_dict['eval_input_fns']
# eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
# predict_input_fn = train_and_eval_dict['predict_input_fn']
# train_steps = train_and_eval_dict['train_steps']

def train_input_fn_az(*args, **kwargs):
    dataset = train_input_fn(*args, **kwargs)
    def remove_bool(x, y):  ## currently does not support bool type, they are not used anyway
        removed_keys = []
        for key, value in x.items():
            if value.dtype == tf.bool:
                removed_keys.append(key)
        for key in removed_keys:
            x.pop(key)
        removed_keys = []
        for key, value in y.items():
            if value.dtype == tf.bool:
                removed_keys.append(key)
        for key in removed_keys:
            y.pop(key)
        return x, y
    dataset = dataset.map(remove_bool)
    dataset = TFDataset.from_tf_data_dataset(dataset,
                                             batch_size=global_batch_size,
                                             hard_code_batch_size=True)
    return dataset

zoo_estimator = TFEstimator(estimator)
zoo_estimator.train(train_input_fn_az, steps=100)