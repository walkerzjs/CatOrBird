import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import os
from copy import deepcopy
from werkzeug.utils import secure_filename
import cv2

model = load_model("./model_CatOrBird_96")
# model.summary()


def createFileList(Mydir, format='.jpg'):
    i = 0
    fileList = []
    breedCount = 0
    for breed in os.listdir(Mydir):
        if (breed == ".DS_Store"):
            continue

        if (breedCount > 3):
            break

        counts = len(os.listdir(Mydir + breed))
        print(counts)
        #        if counts >= 4000 and counts < 50000:
        #        if counts >= 100 and counts<=200:
        if counts >= 0:
            breedCount += 1
            i += 1
            for name in os.listdir(Mydir + breed):
                #                print(name)
                if name.endswith(format):
                    fullName = os.path.join("./validation/", breed, name)
                    #                    print(fullName)
                    fileList.append(fullName)
    return fileList, i


# breeds = "./validation/"
# myFileList, i = createFileList(breeds)
# print(i)
# print(myFileList)
#
# data = []
# labels = []

# loop over the input images
# for image_file in myFileList:
#     # Load the image
#     image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
#
#     # Resize the image so it fits in a 300x300 pixel box
#     dim = (224, 224)
#     image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#
#     # Grab the name of the label based on the folder it was in
#     label = image_file.split(os.path.sep)[-2]
#
#     # Add the  image and it's label to our training data
#     data.append(image)
#     labels.append(label)
#
#
# print("finished")


#
# data = np.array(data, dtype="float") / 255.0
# labels = np.array(labels)
#
# print(np.shape(data))
# print(np.shape(labels))
#
# lb = LabelBinarizer().fit(labels)
# Y_test = lb.transform(labels)


# def start_predict():
#     print("start prediction...")
#     # predictions = model.predict(data)
#     # print(predictions)
#     eval_result = model.evaluate(data, Y_test)
#     print(eval_result)
#     return "Accuracy: {}".format(eval_result[1])

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
    cv2.imwrite(os.path.join("./images/", "{}_{}".format(predictions[0][0],filename)), image_cv)
    if predictions[0][0] >= 0.5:
        result = "cat"
    else:
        result = "bird"

    gap = abs(0.5-predictions[0][0])
    if gap >= 0.4:
        message = "I am very sure this is a photo of {}".format(result)
    elif gap >= 0.2:
        message = "Hmm... I think it is a photo of {}".format(result)
    elif gap >0.1:
        message = "Well... I may be wrong, but is that a photo of {}?".format(result)
    else:
        message = "This is difficult.. You really got me. This is probably a photo of {}".format(result)
    return predictions[0][0], message
