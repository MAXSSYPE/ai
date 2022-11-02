import tensorflow

tensorflow.compat.v1.disable_v2_behavior()


def main():
    x = tensorflow.compat.v1.constant(12.123, dtype=float)
    session = tensorflow.compat.v1.Session()
    print(session.run(x))


if __name__ == '__main__':
    main()
