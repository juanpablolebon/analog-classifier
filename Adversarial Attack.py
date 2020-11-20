from test import *


def fgsm_noise(rgb_data, label):
    image = tf.convert_to_tensor(rgb_data, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)
    gradient = tape.gradient(loss, image)

    return tf.sign(gradient)


noise = fgsm_noise(rgb_sample, 1).numpy()
adversarial_example = rgb_sample + noise * 0.1
perturbed_dataframe = pd.DataFrame(adversarial_example, columns=range(108)).astype(int)
print("Model prediction with original image: ", predict(rgb_sample.astype(int), model), "\n")
print("Model prediction with perturbed image: ", predict(perturbed_dataframe, model))
