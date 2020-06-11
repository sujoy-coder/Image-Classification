from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob

#visual recognition
import json
from watson_developer_cloud import VisualRecognitionV3

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

cf_port = os.getenv("PORT")

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        visual_recognition = VisualRecognitionV3('2018-03-19',iam_apikey='VOvZFOP5_ApeO-IoCqXhh-pqonte77nWUvTBsK-3f4Bu')
        with open(file_path, 'rb') as images_file:
            classes = visual_recognition.classify(images_file,threshold='0.6',classifier_ids='ClassificationModel_1874427257').get_result()
            a=json.loads(json.dumps(classes, indent=2))
            preds=a['images'][0]['classifiers'][0]['classes'][0]['class']
        return preds
    return None


if __name__ == '__main__':
	if cf_port is None:
		app.run(host='0.0.0.0', port=5000, debug=True)
	else:
		app.run(host='0.0.0.0', port=int(cf_port), debug=True)
