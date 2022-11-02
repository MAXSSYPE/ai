import tensorflow

tensorflow.compat.v1.disable_v2_behavior()


def main():
    x = tensorflow.compat.v1.placeholder("float", None)
    function = x + 10
    session = tensorflow.compat.v1.Session()
    print(session.run(function, feed_dict={x: [5.1, 10.2, 15.3, 20.4]}))


if __name__ == '__main__':
    main()
