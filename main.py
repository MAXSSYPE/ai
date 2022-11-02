import tensorflow

tensorflow.compat.v1.disable_v2_behavior()


def main():
    zeros = tensorflow.compat.v1.zeros([5, 6])
    ones = tensorflow.compat.v1.ones([2, 1])
    fill = tensorflow.compat.v1.fill([2, 2], -0.1)
    diag = tensorflow.compat.v1.diag([2, 2, 0.1])
    const = tensorflow.compat.v1.constant([[2, 2], [2, 3]])
    session = tensorflow.compat.v1.Session()
    print("Zeros:")
    print(session.run(zeros))
    print("Ones:")
    print(session.run(ones))
    print("Fill:")
    print(session.run(fill))
    print("Diag:")
    print(session.run(diag))
    print("Const:")
    print(session.run(const))


if __name__ == '__main__':
    main()
