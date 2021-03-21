from flask import Flask,jsonify,request
import json
import numpy as np
from PIL import Image
import base64
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
import io
import PIL.Image



response=" "
app = Flask(__name__)


@app.route("/name",methods =["GET","POST"])
def index():
    global response
    if(request.method=="POST"):
        request_data = request.data #getting the response data
        request_data = json.loads(request_data.decode('utf-8')) #converting it from json to key value pair
        picUrl = request_data["picUrl"] #assigning it to name
        response = picUrl #re-assigning response with the name we got from the user
        return " " #to avoid a type error
    else:
        s=run()
        return jsonify({"DominantColors" : s})
    



def run():
    if(response[0:5]=="https"):
        image = imutils.url_to_image(response)
    else:
        imgdata = base64.b64decode(response)
        print(response)
        imgOne=PIL.Image.open(io.BytesIO(imgdata))
        image=cv2.cvtColor(np.array(imgOne), cv2.COLOR_BGR2RGB)
        


    image = imutils.resize(image, width=250)


  #  skin = extractSkin(image)

 


if __name__ == "__main__":
    app.run(debug=True,host='192.168.43.223')