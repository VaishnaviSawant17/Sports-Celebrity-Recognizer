#install PyWaveletes , numpy, opencv-python, joblib

from flask import Flask, request, jsonify
import util


app = Flask(__name__)

@app.route("/classify_image", methods=["GET","POST"])
def classify_image():
    image_data = request.form['image_data']

    response = jsonify(util.classify_image(image_data)) # to convert data into json format
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__=="__main__":
    print("Starting python Flask Server for Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    app.run(port=5000)

