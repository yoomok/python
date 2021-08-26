from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
import cv2
from PIL import ImageGrab, Image
#from keras.models import load_model
#from keras.preprocessing.image import img_to_array
import numpy as np

app = Flask(__name__)
video_stream = VideoCamera()
@app.route('/')
def index():
    return render_template('index.html')
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/after', methods=['Get','POST'])
def after():
    img = ImageGrab.grab()
    #imgCrop = img.crop((100,100,2000,2000))
    img.save('static/file.png')

    
##########################################
    img1=cv2.imread('static/file.png')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces=cascade.detectMultiScale(gray,1.1,3)

    for x,y,w,h in faces:
        cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)

        cropped=img1[y:y+h,x:x+w]
    cv2.imwrite('static/after.png',img1)
    try:
        cv2.imwrite('static/cropped.png',cropped)
    except:
        pass
##########################################
    try:
        image=cv2.imread('static/cropped.png',0)
    except:   
        image=cv2.imread('static/file.png',0)

    image=cv2.resize(image, (64,64))

    image=image/255.0

    image=np.reshape(image,(1,64,64,1))

    model= load_model('_mini_XCEPTION.102-0.66.hdf5')

    prediction=model.predict(image)

    label_map =["화남" ,"역겨움","무서움", "행복함", "슬픔", "놀람", "무표정"]
    label = ["화가 많이 나셨나봐요.. 화를 풀 수 있는 노래를 추천해 드릴게요.", "언짢으신 일이 있으신가요? 기분 풀 수 있는 노래를 추천해 드릴게요.", "많이 놀라셨나요? 진정할 수 있게 차분한 노래를 추천해 드릴게요.", "행복해보이시네요! 즐거운 노래를 계속 들어보세요!" ,"슬퍼보이시네요. 위로가 될 수 있는 노래를 추천해드릴게요.", "많이 놀라셨나요? 진정할 수 있게 차분한 노래를 추천해 드릴게요." ,"특별한 감정이 판단되지 않았어요. 멜론차트를 들려드릴게요!"]

    prediction=np.argmax(prediction)

    final_prediction=label_map[prediction]

    resultMessage = label[prediction]

    return render_template('after.html', data=final_prediction,message=resultMessage)


if __name__ == '__main__':
	app.run(host='127.0.0.1', debug=True,port="5000")