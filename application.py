from flask import Flask, request
import json
# from catOrBird import start_predict
from catOrBird import processing_image
from flask_cors import CORS

# EB looks for an 'application' callable by default.
application = Flask(__name__)
CORS(application)

@application.route('/catOrBird',methods=['POST'])
def hello_world():
    proba, message = processing_image(request)
    result = {"probability": str(proba), "message": message}
    return json.dumps(result)


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = False
    application.run()