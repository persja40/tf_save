import tensorflow as tf

sess = tf.Session()

hi = tf.constant("Hi Hello TF")
print(sess.run(hi))

a= tf.constant(10)
b= tf.constant(5)
print("a + b = {0}".format(sess.run(a+b)))
