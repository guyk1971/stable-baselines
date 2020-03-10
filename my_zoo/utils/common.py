import os
import warnings
import tensorflow as tf
import logging


def suppress_tensorflow_warnings():
    #----------- Supress Tensorflow version warnings----------------------
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)

    tf.get_logger().setLevel(logging.ERROR)
    #-----------------------------------------------------------------------
    return


def set_gpu_device(gpuid='0'):
    '''

    :param gpuid: id of the device. '0', '1' ...
    to work on the cpu, use empty string ('')
    :return:
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if int(gpuid)>=0 and gpus:    # i.e. we're using a gpu
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Restrict TensorFlow to only use the first GPU
            tf.config.experimental.set_visible_devices(gpus[int(gpuid)], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        cpus = tf.config.experimental.list_physical_devices('CPU')
        tf.config.experimental.set_visible_devices(cpus[0],'CPU')
        print("working on CPU !")
    return

def title(msg,n,ch='='):
    return "\n\n"+ch*n+" "+msg+" "+ch*n


