import os
import cv2


def process_image(image_path, image_save_location, image_shape):
    # first we read the image as greyscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # next we resize the image to our desired resolution of 48x48
    resized_image = cv2.resize(img, image_shape)

    # finally we save the image in specified location
    cv2.imwrite(image_save_location, resized_image)


def preprocess_images(image_source_location, image_shape):
    # we remove the last element from the tuple of shape which is a '1' and is not necessary here
    image_shape = image_shape[:-1]

    image_source_location = image_source_location
    image_destination_location = "{}_{}x{}/".format(image_source_location[:-1], image_shape[0], image_shape[1])

    if not os.path.exists(image_destination_location):
        os.makedirs(image_destination_location)
    else:
        print("Destination location: {} already exists. Preprocessing skipped!!!".format(image_destination_location))
        return image_destination_location

    total_number_images = len(os.listdir(image_source_location))
    success_image_ctr = 0
    failure_image_ctr = 0

    for image in os.listdir(image_source_location):
        image_source_path = os.path.join(image_source_location, image)
        image_dest_path = os.path.join(image_destination_location, image)

        try:
            process_image(image_source_path, image_dest_path, image_shape)
            success_image_ctr += 1
            if success_image_ctr % 1000 == 0:
                print("Processed {}/{} images...".format(success_image_ctr, total_number_images))
        except Exception as e:
            failure_image_ctr += 1

    print("Preprocessing finished. Successfully processed {}/{} images in destination location: {}".
          format(success_image_ctr, total_number_images, image_destination_location))

    return image_destination_location