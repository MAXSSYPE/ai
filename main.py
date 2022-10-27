import tensorflow


def main():
    x = tensorflow.constant(12, dtype=float)
    session = tensorflow.Session()
    print(session.run(x))


if __name__ == '__main__':
    main()
