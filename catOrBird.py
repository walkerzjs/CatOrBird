import numpy as np
from keras.models import load_model
import os
from werkzeug.utils import secure_filename
import cv2

model = load_model("./model_CatOrBird_96")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def processing_image(request):

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    image = request.files['image']
    filename = secure_filename(image.filename)
    if not allowed_file(image.filename):
        raise Exception("Sorry, I can't read this photo. I can only read .png, .jpg and .jpeg formats currently.")
    # read image using openCV
    image_cv = np.asarray(bytearray(image.read()), dtype="uint8")
    image_cv = cv2.imdecode(image_cv, cv2.IMREAD_COLOR)
    dim = (224, 224)
    image_cv = cv2.resize(image_cv, dim, interpolation=cv2.INTER_AREA)
    data = [image_cv]
    data = np.array(data, dtype="float") / 255.0
    predictions = model.predict(data)
    # cv2.imwrite(os.path.join("./images/", "{}_{}".format(predictions[0][0],filename)), image_cv)
    if predictions[0][0] >= 0.5:
        result = "cat"
    else:
        result = "bird"

    gap = abs(0.5-predictions[0][0])
    if gap >= 0.4:
        message = "I am pretty sure this is a photo of {}.".format(result)
    elif gap >= 0.2:
        message = "Hmm... I think it is a photo of {}.".format(result)
    elif gap >0.1:
        message = "Well... I may be wrong, but is that a {}?".format(result)
    else:
        message = "This is difficult.. You really got me. This is probably a photo of {}.".format(result)
    return predictions[0][0], message
