import numpy as np
import cv2
import click
import keras.backend as K
from keras.models import load_model
from deepctr.layers import custom_objects
from const import colors

# metrics
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

class Tracker(object):
    
    def __init__(self, model, threshold, rec, video_file):
        custom_objects["dice_coef"] = dice_coef
        self.model = load_model(model, custom_objects=custom_objects)
        self.class_nb = self.model.layers[-1].output_shape[-1]
        self.input_shape = self.model.layers[0].input_shape[1:]
        if video_file == "":
            video_file = 0
        self.cap = cv2.VideoCapture(video_file)
        self.rec = rec
        self.out = None
        if self.rec != "":
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(rec , fourcc, 10.0, (self.input_shape[1], self.input_shape[0]))
        self.threshold = threshold
        
    def loop(self):
        while(True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            img = cv2.resize(frame,(self.input_shape[1], self.input_shape[0]))
            # Operations on the frame
            img = np.array(img) / 255.0
            pred = self.model.predict(img.reshape((1, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
            for j in range(self.class_nb):
                msk = pred[0,:,:,j]
                img[msk>=self.threshold] = [c/255. for c in colors[j]]
            # Display the resulting frame
            cv2.imshow('frame',img)
            if self.rec != "":
                self.out.write(img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
    def stop(self):
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()

@click.command()
@click.argument("model", default="model.h5")
@click.option("-v", "video_file", default="", help="video file to track")
@click.option("-r", "rec", default="", help="record video to file")
@click.option("-t", "threshold", default=0.9, help="detection threshold")
def main(model, threshold, rec, video_file):
    tracker = Tracker(model, threshold, rec, video_file)
    tracker.loop()
    tracker.stop()

if __name__ == "__main__":
    main()
