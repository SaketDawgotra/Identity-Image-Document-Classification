import os
from flask import Flask, render_template, request, send_from_directory
from datetime import datetime
import datetime
import time

start=time.time()
#need based
timestamp = 1545730073
#dt_object = datetime.fromtimestamp(timestamp)
__author__ = 'saket'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# check lib env var

import cv2
import numpy as np
from deepgaze.color_classification import HistogramColorClassifier

my_classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128],
                                         hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')
model_1 = cv2.imread('train_images/panx.jpeg')  # pan
model_2 = cv2.imread('train_images/pt12.png')  # passport
model_3 = cv2.imread('train_images/papa_pass.jpeg')  # passport sp case 1
model_4 = cv2.imread('train_images/pass9.jpg')  # pasport sp case 2
model_5 = cv2.imread('train_images/tm000070.jpg')  # pan sp case 1

my_classifier.addModelHistogram(model_1)
my_classifier.addModelHistogram(model_2)
my_classifier.addModelHistogram(model_3)
my_classifier.addModelHistogram(model_4)
my_classifier.addModelHistogram(model_5)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

@app.route("/")
def index():
    return render_template("upload.html")


def allowed_file(filename):
    return not (not ('.' in filename) or not (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS))

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, "images/")
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    file = request.files['file']
    if file and allowed_file(file.filename):
        for upload in request.files.getlist("file"):
            print(upload)
            print("{} is the file name".format(upload.filename))
            filename = upload.filename
            destination = "/".join([target, filename])
            print ("Accept incoming file:", filename)
            print ("Save it to:", destination)
            upload.save(destination)

            image2 = cv2.imread(destination)

            comparison_array1 = my_classifier.returnHistogramComparisonArray(image2, method="intersection")

           # print(comparison_array1)

            comparison_distribution1 = comparison_array1 / np.sum(comparison_array1)

            if max(comparison_distribution1) == comparison_distribution1[1] or max(comparison_distribution1) == \
                    comparison_distribution1[2] or max(comparison_distribution1) == comparison_distribution1[3]:
                text = "PASSPORT"
            elif max(comparison_distribution1) == comparison_distribution1[0] or max(comparison_distribution1) == \
                    comparison_distribution1[4]:
                text = "PAN"
           # value=max(comparison_distribution1)

            end = time.time()
            print("time of completion:", end - start)

            print(comparison_distribution1)
            f = open("results.txt", "a+").write(filename)
            f = open("results.txt", "a+").write(" , ")
            f = open("results.txt", "a+").write(text)
          #  f = open("results.txt", "a+").write(",")
            #f = open("results.txt", "a+").write(dt_object)
            f = open("results.txt", "a+").write("\t")
            f = open("results.txt", "a+")
            f.write(datetime.datetime.now().ctime())
            f.close()
            f = open("results.txt", "a+").write("\n")

        # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", image_name=filename, pred=text)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0')

