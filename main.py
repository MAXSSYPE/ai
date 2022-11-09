import tensorflow

tensorflow.compat.v1.disable_v2_behavior()


def main():
    x = tensorflow.compat.v1.placeholder(tensorflow.compat.v1.int32, [5])
    y = tensorflow.compat.v1.placeholder(tensorflow.compat.v1.int32, [5])
    acc, acc_op = tensorflow.compat.v1.metrics.accuracy(labels=x, predictions=y)
    global_init = tensorflow.compat.v1.global_variables_initializer()
    local_init = tensorflow.compat.v1.local_variables_initializer()
    session = tensorflow.compat.v1.Session()
    session.run(global_init)
    session.run(local_init)
    val = session.run([acc, acc_op], feed_dict={x: [1, 1, 0, 1, 0], y: [0, 1, 0, 0, 1]})
    val_acc = session.run(acc)
    print(val_acc)


if __name__ == '__main__':
    main()
