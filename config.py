from imports import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)

digital_rgb_data_dir = r'C:/Users/juanp/Documents/Images/RGBDigFace.csv'
film_rgb_data_dir = r'C:/Users/juanp/Documents/Images/RGBFilmFace.csv'
combined_rgb_data_dir = r'C:/Users/juanp/Documents/Images/RGBCombFace.csv'
digital_images_dir = r'C:/Users/juanp/Documents/Images/Faces_Dig'
film_images_dir = r'C:/Users/juanp/Documents/Images/Faces_Film'
model_dir = r'C:/Users/juanp/Documents/Images/Model'
model = keras.models.load_model(r'C:/Users/juanp/Documents/Images/Model')
