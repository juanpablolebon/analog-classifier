from collector import *


def combine_csv():
    film_face = pd.read_csv(film_rgb_data_dir)
    film_face['Dig'] = 0
    dig_face = pd.read_csv(digital_rgb_data_dir)
    dig_face['Dig'] = 1
    comb = pd.concat(([film_face, dig_face])).astype(int).reset_index(drop=True)
    comb.to_csv(combined_rgb_data_dir, index=False)


def convolve(scrape_, is_dig, grid_size):
    if scrape_:
        if is_dig:
            scrape(0, 0, grid_size, 0.05)
        else:
            scrape(0, 1, grid_size, 0.05)
    combine_csv()
    return 1


size = 6
data = pd.read_csv(combined_rgb_data_dir)
print(data)
data_columns = data.columns
predictors = data[data_columns[data_columns != 'Dig']]
target = data['Dig']
n_cols = (size ** 2) * 3


def build_model(cols):
    new_model = Sequential()
    new_model.add(Dense(15, activation='relu', input_shape=(cols,)))
    for i in range(19):
        new_model.add(Dense(15, activation='relu'))
    new_model.add(Dense(2, activation='softmax'))
    new_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse',
                      metrics=['accuracy'])
    return new_model


record = 0.0845


def train(lowest_mse):
    x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=30,
                                                        shuffle=True)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    mse_prev = lowest_mse
    count = 0
    for i in range(20):
        fit = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=1)
        print(fit)
        print("\n Testing:")
        mse = model.evaluate(x_test, y_test, verbose=1)[0]
        if mse < mse_prev:
            count += 1
            print("\nRecord low MSE! decreased by", mse_prev - mse, "-", count, "so far.\n")
            model.save(model_dir)
            mse_prev = mse


train(record)
