
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2


def capture_live_img():
    #Initializing Picam instance with img resolution and frame
    camera_instance = PiCamera()
    camera_instance.resolution = (640, 480)
    camera_instance.framerate = 32
    raw_image = PiRGBArray(camera_instance, size=(740, 480))

    time.sleep(0.5)

    # Taking frames from the picamera
    img_list=[]
    for frame in camera_instance.capture_continuous(raw_image, format="bgr", use_video_port=True):
        image = frame.array
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1)
        img_list.append(image)
        raw_image.truncate(0)

    return img_list

def write_img(img_list,file_path)
    if len(img_list)>0:
        counter=0
        for img in img_list:
            counter+=1
            cv2.imwrite(file_path+'/'+counter+'.jpg', img)





