
from config import *
from matplotlib import colors


def sample(file, grid_size, stride):
    if file.endswith("png"):
        img = (imread(file, 'RGB') * 255).astype(int)
    else:
        img = imread(file).astype(int)
    empty = pd.DataFrame(columns=(range((grid_size ** 2) * 3)))
    samples = samples_aux = empty
    if img.shape[-1] != 3:
        print("HSL conversion unavailable; no samples were taken")
        return empty
    hsl = matplotlib.colors.rgb_to_hsv(img / 255)
    wid = len(img[0])
    height = len(img)
    overlap = 5  # divides stride values to control the amount of overlap consecutive grids can have
    count = valid = row = k = 0
    start_time = time.time()
    stride_row = (stride / 100) * wid / overlap
    stride_col = (stride / 100) * height / overlap
    possible_samples_per_row = wid * overlap // (grid_size + stride_row))
    possible_samples_per_col = height * overlap // (grid_size + stride_col))

    for rows_done in tqdm(range(possible_samples_per_col)):
        col = 0
        progress = (rows_done // possible_samples_per_col) * 100
        # Write checkpoints every time progress increases by 5%, except at 100% where it would be unnecessary
        # avoids working with large dataframe
        if progress % 5 == 0 and progress != 100:
            k = 0
            samples_aux = pd.concat([samples_aux, samples], ignore_index=True)
            samples = empty
        for cols_done in range(possible_samples_per_row):
            # A simple time saving measure, created due to my prioritizing the amount of images I processed rather than
            # the amount of samples I took per image
            if time.time() - start_time > 600:
                print("Sampling timed out, returning gathered data only")
                return samples_aux  # returns last checkpoint
            L = 0
            for n in range(grid_size):
                for m in range(grid_size):
                    if col + m >= height or row + n >= wid:
                        break
                    pixel = hsl[col + m][row + n]
                    # pixel[1] stores a given pixel's saturation on a scale of 0 to 1. This check simply makes sure
                    # that the image is not black and white
                    if pixel[1] >= 0.1:
                        valid = 1
                    for splitter in range(3):  # maps rgb values to separate cells
                        samples.loc[k, L + splitter] = img[col + m][row + n][splitter]
                    L += 3
            if L > 0:
                k +=1
                count += 1
            col = int(col + grid_size + stride_col)
        row = int(row + grid_size + stride_row)
    samples = pd.concat([samples_aux, samples])
    if valid:
        return samples.dropna().drop_duplicates(subset=None, keep='first', inplace=False)
    else:
        print("Image was black and white; no samples were added\n\n")
        return empty


def crop_center(img, cropx, cropy):
    x = len(img[0])
    y = len(img)
    print(x, y)
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def scrape(create_new, is_dig, grid_size, stride):
    if create_new:
        frame = pd.DataFrame(columns=(range(int((grid_size ** 2) * 3))))
        if is_dig:
            frame.to_csv(digital_rgb_data_dir)
            img_dir = digital_images_dir
        else:
            frame.to_csv(film_rgb_data_dir)
            img_dir = film_images_dir
    else:
        if is_dig:
            frame = pd.read_csv(digital_rgb_data_dir)
            img_dir = digital_images_dir
        else:
            frame = pd.read_csv(film_rgb_data_dir)
            img_dir = film_images_dir

    count = 1

    for filename in os.listdir(img_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            filename = img_dir + "/" + filename
            print("Processing Image", count, "...")
            samples = sample(filename, grid_size, stride)
            samples.columns = frame.columns = range(int((grid_size ** 2) * 3))
            frame = pd.concat([frame, samples])

        if count % 10 == 0:
            frame = frame.dropna().drop_duplicates(subset=None, keep='first', inplace=False).astype(int).reset_index(
                drop=True)
            if is_dig:
                frame.to_csv(digital_rgb_data_dir, index=False)
            else:
                frame.to_csv(film_rgb_data_dir, index=False)
        count += 1
    frame = frame.dropna().drop_duplicates(subset=None, keep='first', inplace=False).astype(int).reset_index(drop=True)
    if is_dig:
        frame.to_csv(digital_rgb_data_dir, index=False)
    else:
        frame.to_csv(film_rgb_data_dir, index=False)

