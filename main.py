import tensorflow
import tensorflow_datasets
import matplotlib.pyplot as plt

tensorflow.compat.v1.disable_v2_behavior()
tensorflow.compat.v1.enable_eager_execution()


def main():
    ds = tensorflow_datasets.load('beans', split='train', shuffle_files=True)
    for index, a in enumerate(ds):
        plt.imshow(a['image'])
        plt.show()
        if index >= 0:
            break

    resized_dataset = ds.map(map_func=augment_hue)
    normalized_dataset = resized_dataset.map(map_func=normalize_image)

    for index, (image, label) in enumerate(resized_dataset):
        plt.imshow(image / 255.0)
        plt.show()
        print(label.numpy())
        if index >= 0:
            break

    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Input(shape=(62, 62, 3)),
        tensorflow.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        tensorflow.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        tensorflow.keras.layers.GlobalMaxPooling2D(),
        tensorflow.keras.layers.Dense(32, activation='relu'),
        tensorflow.keras.layers.Dense(3, activation='softmax')
    ])

    # Compile and train the model for one epoch... It's only to have something trained, not get the best score
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tensorflow.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )

    batch = 8
    dataset = normalized_dataset.repeat().batch(batch)

    model.fit(dataset, steps_per_epoch=(len(ds) / batch), epochs=3)

    test_dataset = tensorflow_datasets.load('beans', split='test')
    test_dataset = test_dataset.map(map_func=augment_hue).map(map_func=normalize_image).batch(batch)
    score = model.evaluate(test_dataset, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


def augment_hue(tensor):
    print(tensor.keys())
    return tensorflow.image.resize(tensor['image'], (62, 62)), tensor.get('label', tensor.get('labels'))


def normalize_image(image, label):
    return image / 255.0, label


if __name__ == '__main__':
    main()
