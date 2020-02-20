import sys
import logging
from my_zoo.utils.common import MyLogger

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
#     filename="out.log",
#     filemode='a'
# )



stdout_logger = MyLogger('STDOUT',"my_out.log").get_logger()
sl = StreamToLogger(stdout_logger, logging.INFO)
sys.stdout = sl

# stderr_logger = logging.getLogger('STDERR')
sl = StreamToLogger(stdout_logger, logging.ERROR)
sys.stderr = sl

print("Test to standard out")
raise Exception('Test to standard error')


N=10
n_a=5
tf.reset_default_graph()
gw_logits=tf.random.uniform(shape=(N,n_a),minval=-10,maxval=10)
tau=tf.placeholder(tf.float32,shape=())
gw_max=tf.reduce_max(gw_logits,axis=1,keepdims=True)
gw_norm=tf.math.subtract(gw_logits,gw_max)
gw_masked=tf.where(gw_norm>tf.math.log(tau),gw_logits,tf.constant(-np.inf)*tf.ones_like(gw_logits))

S=tf.Session()
gwl_vals,gwm_vals=S.run([gw_logits,gw_masked],feed_dict={tau:0.3})