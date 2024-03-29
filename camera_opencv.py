import cv2
from base_camera import BaseCamera
import ocr

class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        process_stream(camera)
       
        #yield cv2.imencode('.jpg', img)[1].tobytes()
