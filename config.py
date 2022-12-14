from absl import app
from absl import flags


flags.DEFINE_bool('train', False, 'do training')
flags.DEFINE_bool('test', False, 'do testing')

flags.DEFINE_integer('seed', 0, 'random seed')
flags.DEFINE_integer('ngpu', 1, 'number of gpus to use')
flags.DEFINE_integer('local_rank', 0, 'for distributed training')
flags.DEFINE_integer('num_workers', 8, 'dataset workers')

flags.DEFINE_string('checkpoint_dir', 'log', 'Root directory for output files')
flags.DEFINE_string('name', 'exp', 'Experiment Name')

flags.DEFINE_string('train_list', '', 'name of the training videos')
flags.DEFINE_string('test_list', '', 'name of the testing videos')
flags.DEFINE_string('model_path', '', 'load model path')
flags.DEFINE_string('vis_path', '', 'the visualization dir')

flags.DEFINE_integer('total_iters', 10000, 'number of training iterations')
flags.DEFINE_integer('batch_log_interval', 10, 'log interval of batches (iterations)')
flags.DEFINE_integer('save_freq', 1, 'save model every k iters')
flags.DEFINE_integer('vis_freq', 1, 'visualize training every k iters during training')
flags.DEFINE_integer('batch_size', 4, 'Size of minibatches')
flags.DEFINE_integer('dframe_eval', 1, 'dframe when predicting')
flags.DEFINE_string('logger', 'tensorboard', 'logger (tensorboard or wandb)')

opts = flags.FLAGS
