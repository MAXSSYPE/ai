import tensorflow

tensorflow.compat.v1.disable_v2_behavior()


def main():
    tgh = tensorflow.compat.v1.nn.tanh(
        [-1.0, -0.99990916, -0.46211717, 0.7615942, 0.8336547, 0.9640276, 0.9950547, 1.0])
    sigmoid = tensorflow.compat.v1.nn.sigmoid([5.0, 6.0, 7.0, 8.0, 9.0, -10.0])
    relu = tensorflow.compat.v1.nn.relu([-2.0, 0.0, 3.0])
    elu = tensorflow.compat.v1.nn.elu(-1000.0)
    session = tensorflow.compat.v1.Session()
    print("tgh:")
    print(session.run(tgh))
    print("sigmoid:")
    print(session.run(sigmoid))
    print("relu:")
    print(session.run(relu))
    print("elu:")
    print(session.run(elu))


if __name__ == '__main__':
    main()
