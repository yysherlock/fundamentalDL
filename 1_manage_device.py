import tensorflow as tf

with tf.device('/gpu:0'):
    a = tf.constant([1.0,2.0,3.0,4.0], shape=[2,2], name='a')
    b = tf.constant([1.0,2.0], shape=[2,1], name='b')
    c = tf.matmul(a,b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(c)
sess.close()
