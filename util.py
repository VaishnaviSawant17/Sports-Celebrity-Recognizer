import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import os

__class_name_to_number = {}
__class_number_to_name = {}
__model = None  # private variable initialize to none


def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)  # clear face and 2 eyes give this array

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))  # convert all raw images in one size
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))  # convert all wavelet transform image in one size
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1),
                                  scalled_img_har.reshape(32 * 32, 1)))  # vstack = to vertical stacking of images
        # (32 * 32, 1) this image is black and white so it does not have 3rd dimension
        # (32 * 32 * 3, 1) in this 3rd dimension is rgb

        len_image_array = 32 * 32 * 3 + 32 * 32

        # some of the API expect float datatype so we have to convert
        final = combined_img.reshape(1, len_image_array).astype(float)
        # predict = most of predict function need more than one image so we can't directly return

        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            # name of person in output instead number given in output
            'class_probability': np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })
    return result


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


def load_saved_artifacts():  # call this method in main function
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    current_dir = os.path.dirname(__file__)
    artifact_path = os.path.join(current_dir, 'artifacts', 'class_dictionary.json')
    with open(artifact_path, 'r') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        current_dir = os.path.dirname(__file__)
        artifact_path = os.path.join(current_dir,'artifacts','saved_model.pkl')
        with open(artifact_path ,'rb') as f:
            __model = joblib.load(f)  # open file and loading it using joblib module
    print("loading saved artifacts...done")


# taking base64 string and return in cv2 image
# using numpy and cv2 imdecode function to convert into open CV image
def get_cv2_image_from_base64_string(b64str):
    # Check if the string contains a comma, indicating the presence of a prefix
    if ',' in b64str:
        encoded_data = b64str.split(',')[1]
    else:
        # If no comma, assume the entire string is base64 data
        encoded_data = b64str
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# if get_cropped_image_if_2_eyes function not detect anything then it gives empty array
def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('Server/opencv'
                                         '/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('Server/opencv'
                                        '/haarcascades/haarcascade_eye.xml')
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()  # call method

    # print(classify_image(get_b64_test_image_for_virat(), None))

    # print(class_number_to_name(4))
    print(classify_image(None,
                         "C:\\Users\\SUNIL SAWANT\\Desktop\\AIMLpractsem2\\MachineLearning "
                         "krish\\SportPersonClassifier\\text_image\\federer1.jpg"))
    print(classify_image(None,
                         "C:\\Users\SUNIL SAWANT\\Desktop\\AIMLpractsem2\\MachineLearning "
                         "krish\\SportPersonClassifier\\text_image\\federer2.jpg"))
    print(classify_image(None,
                         "C:\\Users\\SUNIL SAWANT\\Desktop\\AIMLpractsem2\\MachineLearning "
                         "krish\\SportPersonClassifier\\text_image\\virat1.jpg"))
    print(classify_image(None,
                         "C:\\Users\\SUNIL SAWANT\\Desktop\\AIMLpractsem2\\MachineLearning "
                         "krish\\SportPersonClassifier\\text_image\\virat2.jpg"))
    print(classify_image(None, "C:\\Users\\SUNIL SAWANT\\Desktop\\AIMLpractsem2\\MachineLearning "
                               "krish\\SportPersonClassifier\\text_image\\virat3.jpg"))
    # Inconsistent result could be due to https://github.com/scikit-learn/scikit-learn/issues/13211
