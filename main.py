import tensorflow

tensorflow.compat.v1.disable_v2_behavior()


def main():
    x = tensorflow.compat.v1.constant(12.123, dtype=float)
    y = tensorflow.compat.v1.Variable(x + 12.123)
    model = tensorflow.compat.v1.global_variables_initializer()
    session = tensorflow.compat.v1.Session()
    session.run(model)
    print(session.run(y))


if __name__ == '__main__':
    main()
