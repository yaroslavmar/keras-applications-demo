from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from PIL import ImageDraw

import numpy as np
import sys

def classify_image(path_image, num_top_classes=3):
    model = ResNet50(weights='imagenet')

    img = image.load_img(path_image, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    return img, decode_predictions(preds, top=num_top_classes)[0]

#img_path = 'images/car.jpeg'

if __name__ == '__main__':
    path_image = sys.argv[1]
    img, predicted_image = classify_image(path_image)
    predicted_text = [str(x[1:]) + '\n' for x in predicted_image]
    predicted_text = "".join(predicted_text)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), predicted_text, (255,99,71))
    img.save(path_image.replace('images', 'images_predicted'))
