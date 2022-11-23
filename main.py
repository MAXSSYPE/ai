from __future__ import print_function
import tensorflow
import tensorflow_datasets
import mnist
import numpy


tensorflow.compat.v1.disable_v2_behavior()


def main():
    batch_size = 128
    num_classes = 10
    epochs = 2
    x_train, y_train, x_test, y_test = mnist.fashion_mnist()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = (x_train - numpy.mean(x_train)) / numpy.std(x_train)
    x_test = (x_test - numpy.mean(x_test)) / numpy.std(x_test)

    y_train = tensorflow.compat.v1.keras.utils.to_categorical(y_train, num_classes)
    y_test = tensorflow.compat.v1.keras.utils.to_categorical(y_test, num_classes)

    model = tensorflow.compat.v1.keras.models.Sequential()
    model.add(tensorflow.compat.v1.keras.layers.Dense(256, activation='elu', input_shape=(784,)))
    model.add(tensorflow.compat.v1.keras.layers.Dropout(0.4))
    model.add(tensorflow.compat.v1.keras.layers.Dense(512, activation='relu'))
    model.add(tensorflow.compat.v1.keras.layers.Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    model.save('my_model.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


if __name__ == '__main__':
    main()
