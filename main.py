import tensorflow
import tensorflow_datasets
import matplotlib.pyplot as plt

tensorflow.compat.v1.disable_v2_behavior()
tensorflow.compat.v1.enable_eager_execution()


def main():
    ds = tensorflow_datasets.load('voc/2007', split='train', shuffle_files=True)
    for index, a in enumerate(ds):
        plt.imshow(a['image'])
        plt.show()
        print(a.keys())
        print(a['labels'].numpy())
        if index >= 0:
            break

    def augment_hue(tensor):
        print(tensor.keys())
        return tensorflow.image.resize(tensor['image'], (62, 62)), tensor.get('label', tensor.get('labels'))

    def normalize_image(image, label):
        return image / 255.0, label

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

    test_dataset = tensorflow_datasets.load('beans', split='test')
    test_dataset = test_dataset.map(map_func=augment_hue).map(map_func=normalize_image).batch(8)
    print(model.evaluate(test_dataset))


if __name__ == '__main__':
    main()
