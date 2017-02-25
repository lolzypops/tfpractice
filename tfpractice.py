import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data', one_hot=True)

# Construction Phase
x = tf.placeholder(tf.float32, shape=[None, 784]) # placeholder for the input
y = tf.placeholder(tf.float32, shape=[None, 10]) # placeholder for label

with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
    h = tf.sigmoid(tf.matmul(x, W1) + b1)

with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
    y_predict = tf.matmul(h, W2) + b2

with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict)) # Cross entropy loss
    tf.summary.scalar('loss', cross_entropy)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("training"):
    backprop = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("graphs", sess.graph)
sess.run(tf.global_variables_initializer())

train_step = 2000
batch_size = 50

for i in range(train_step):
    batch_x, batch_y = data.train.next_batch(batch_size) # next_batch grabs batch of size 50 (random), creates a tensor of size(50, 784)
    sess.run(backprop, feed_dict={x:batch_x, y:batch_y})
    current_loss = sess.run(cross_entropy, feed_dict={x:batch_x, y:batch_y})
    if i % 50 == 0:
        summary = sess.run([merged], feed_dict={x:batch_x, y:batch_y})
        test_accuracy = sess.run(accuracy, feed_dict={x:data.test.images, y:data.test.labels})
        print("Iter {}: {}".format(i, test_accuracy))
