# Ignore this library

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img



def balance_data(use_original_only:bool, csv_name:str):
    """Clones transformed data, there is an even number of samples with different labels.
    Appends cloned data information to data.csv"""
    counter = DATA['rays'].value_counts()
    label_counter = {}

    # For each relevant label, count appearances.
    for lbl in counter.keys():
        if lbl in LABELS:
            label_counter[lbl] = counter[lbl]

    most_common = max(label_counter)
    least_common = min(label_counter)

    # Select group to clone
    kernel = []
    for i, label in enumerate(DATA_LABEL):
        fname = DATA_FILENAME[i]
        relevant = label == least_common and not (use_original_only and 'aug' in str(fname))
        if relevant: kernel.append(fname)

    # For each image, randomly rotate and move around, then save it with the same csv information
    # Do it until there is the same amount of pictures for both of the classes.
    for k in range(label_counter[most_common] - label_counter[least_common]):
        rnd_id = str(random.sample(kernel, 1)[0]) # Get random sample to clone
        label = get_label(rnd_id)
        label = str(label)

        curr_dir = DATA_PATH + label + "\\"
        image = cv2.imread(curr_dir + rnd_id + ".jpg")
        image = change_brightness(image, value= 40 - random.randint(0, 80))  # increases

        # dividing height and width by 2 to get the center of the image
        height, width = image.shape[:2]
        # get the center coordinates of the image to create the 2D rotation matrix
        center = (width / 2, height / 2)

        # using cv2.getRotationMatrix2D() to get the rotation matrix, with a random angle
        theta = random.randint(0, 360)
        rotate_matrix = cv2.getRotationMatrix2D(center = center, angle = theta, scale = 1)

        # Randomly move image around
        w = int(width / 8)
        h = int(height / 8)
        a = random.randint(- w, w)
        b = random.randint(- h, h)
        trans_matrix = np.float32([[1, 0, a], [0 ,1, b]])

        # transform the image using cv2.warpAffine
        rotated_image = cv2.warpAffine(src = image, M = rotate_matrix, dsize = (width, height))
        trans_image = cv2.warpAffine(src = rotated_image, M = trans_matrix, dsize = (width, height))

        # Show image while changing it.
        # plt.imshow(trans_image)
        # plt.show()

        # Get available name
        curr_index = 1
        available = False
        fnames = os.listdir(curr_dir)
        while not available:

            ## Iterate over [aug_6rays_1, aug_6rays_2, ...] and get the lateast aug_6rays_<x> to save it
            curr_name = 'aug_6rays_' + str(curr_index)
            available = curr_name + '.jpg' not in fnames
            curr_index += 1

        ## Write in data.csv the new clone, not needed if data is loaded directly from directory. ##
        # # Read previous attributes (rays, daidem)
        # with open(PATH + csv_name, newline='') as csvfile:
        #     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #     for row in spamreader:
        #         id, rays, circle = row[0].split(SEP)
        #         if rnd_id == id: # Found relevant row
        #             break
        #
        # # Write new image id with same attributes
        # with open(PATH + csv_name, 'a', newline='') as csvfile:
        #     spamwriter = csv.writer(csvfile, delimiter=SEP,
        #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     spamwriter.writerow([curr_name, rays, circle])

        # Save image in the data\star directory.
        os.chdir(curr_dir)
        cv2.imwrite(curr_name + '.jpg', trans_image)
