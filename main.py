import tensorflow

tensorflow.compat.v1.disable_v2_behavior()


def main():
    x = tensorflow.compat.v1.placeholder("float", [None, None])
    function = x + 10
    session = tensorflow.compat.v1.Session()
    data = [[12.3, 89.1], [123.123, -0.432], [90.1, -90.1]]
    print(session.run(function, feed_dict={x: data}))


if __name__ == '__main__':
    main()
