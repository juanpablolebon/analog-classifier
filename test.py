from collector import *


def predict(df, network):
    pred = network.predict(df)
    length = len(pred)
    val = 0
    for i in range(length):
        if pred[i][0] > pred[i][1]:
            val += 1
        else:
            val -= 1
    print(str(abs(val) * 100 / df.shape[0]) + "% confident")
    return int(val <= 0)


rgb_sample = sample(digital_images_dir + "/59100.png", 6, 50).astype(int)

# predict(rgb_sample, model)
