import tensorflow

tensorflow.compat.v1.disable_v2_behavior()


def main():
    x = tensorflow.compat.v1.Variable(3, dtype=tensorflow.compat.v1.float32)
    log_x = tensorflow.compat.v1.log(x)
    log_x_squared = tensorflow.compat.v1.square(log_x)
    optimizer = tensorflow.compat.v1.train.GradientDescentOptimizer(0.7)
    train = optimizer.minimize(log_x_squared)
    init = tensorflow.compat.v1.global_variables_initializer()
    session = tensorflow.compat.v1.Session()
    session.run(init)
    print("starting at ", "x: ", session.run(x), "log(x)^2: ", session.run(log_x_squared))
    for step in range(10):
        session.run(train)
        print("step: ", step, "x: ", session.run(x), "log(x)^2: ", session.run(log_x_squared))


if __name__ == '__main__':
    main()
