from flask import Flask,jsonify,request
import json
import numpy as np
from PIL import Image
import io
import PIL.Image

response="rat.rat.comdummy"
app = Flask(__name__)


@app.route("/name",methods =["GET","POST"])



if __name__ == "__main__":
    app.run(debug=True,host='192.168.43.223')