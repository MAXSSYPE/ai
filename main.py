import tensorflow

tensorflow.compat.v1.disable_v2_behavior()


def main():
    image = tensorflow.compat.v1.image.decode_png(tensorflow.compat.v1.read_file("test.png"), channels=3)
    session = tensorflow.compat.v1.Session()
    print(session.run(tensorflow.compat.v1.shape(image)))


if __name__ == '__main__':
    main()
