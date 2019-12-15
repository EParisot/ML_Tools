import numpy as np
import cv2
import click
from tensorflow.keras.models import load_model

classes = ["cardboard", "glass", "metal", "paper", "plastic"]

class Tracker(object):
    
    def __init__(self, model, video_file):
        print("Loading %s" % model)
        self.model = load_model(model)
        self.input_shape = self.model.layers[0].input_shape[0][1:]
        print("Loaded model with input shape %s" % str(self.input_shape))
        if video_file == "":
            video_file = 0
        self.cap = cv2.VideoCapture(video_file)


    def loop(self):
        while(True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if ret:
                img = cv2.resize(frame,(self.input_shape[1], self.input_shape[0]))
                # Operations on the frame
                img = np.array(img) / 255.0
                pred = self.model.predict(img.reshape((1, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
                # Display the resulting frame
                cv2.putText(img, classes[np.argmax(pred[0])] ,(20,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
                cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

@click.command()
@click.argument("model", default="model.h5")
@click.option("-v", "video_file", default="", help="video file to track")
def main(model, video_file):
    tracker = Tracker(model, video_file)
    tracker.loop()
    tracker.stop()

if __name__ == "__main__":
    main()
