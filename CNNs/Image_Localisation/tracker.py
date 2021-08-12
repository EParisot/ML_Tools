import numpy as np
import cv2
import click
import tensorflow as tf
from tensorflow.keras.models import load_model
from deepctr.layers import custom_objects
from tensorflow.python.framework.c_api_util import tf_buffer
from const import colors

for i, c in enumerate(colors):
    colors[i] = list(reversed(c))

class Tracker(object):
    
    def __init__(self, model, class_nb, nb_boxes, rec, video_file):
        self.class_nb = class_nb + 1
        self.nb_boxes = nb_boxes
        self.grid_w = 7
        self.grid_h = 7
        custom_objects["custom_loss"] = self.custom_loss
        self.model = load_model(model, custom_objects=custom_objects)
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
        self.img_w = self.input_shape[1]
        self.img_h = self.input_shape[0]
        self.cell_w = self.img_w / self.grid_w
        self.cell_h = self.img_h / self.grid_h

    
    
    def custom_loss(self, y_true, y_pred):

        y_true_class = y_true[...,:self.class_nb]
        y_pred_class = y_pred[...,:self.class_nb]

        pred_boxes = tf.reshape(y_pred[...,self.class_nb:], (-1, self.grid_w * self.grid_h, self.nb_boxes, 5))
        true_boxes = tf.reshape(y_true[...,self.class_nb:], (-1, self.grid_w * self.grid_h, self.nb_boxes, 5))
        
        y_pred_xy   = pred_boxes[...,0:2]
        y_pred_wh   = pred_boxes[...,2:4]
        y_pred_conf = pred_boxes[...,4]

        y_true_xy   = true_boxes[...,0:2]
        y_true_wh   = true_boxes[...,2:4]
        y_true_conf = true_boxes[...,4]

        clss_loss  = tf.math.reduce_sum(tf.math.square(y_true_class - y_pred_class)) #* y_true_conf
        xy_loss    = tf.math.reduce_sum(tf.math.square(y_true_xy - y_pred_xy)) * y_true_conf
        # sqrt on wh to penalize bigger boxes
        wh_loss    = tf.math.reduce_sum(tf.math.square(tf.math.sqrt(y_true_wh) - tf.math.sqrt(y_pred_wh))) * y_true_conf
        #wh_loss    = tf.math.reduce_sum(tf.math.square(y_true_wh - y_pred_wh)) * y_true_conf
        # when we add the confidence the box prediction lower in quality but we gain the estimation of the quality of the box
        # however the training is a bit unstable

        # minimum distance between boxes distance between the two center
        intersect_wh = tf.maximum(tf_buffer.zeros_like(y_pred_wh), (y_pred_wh + y_true_wh)/2 - tf.abs(y_pred_xy - y_true_xy) )
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
        
        true_area  = y_true_wh[...,0] * y_true_wh[...,1]
        pred_area  = y_pred_wh[...,0] * y_pred_wh[...,1]
        union_area = pred_area + true_area - intersect_area
        
        iou = intersect_area / union_area

        conf_loss = tf.math.square(y_true_conf * iou - y_pred_conf) * y_true_conf
        l_coords = 3
        return clss_loss + l_coords * xy_loss + l_coords * wh_loss + conf_loss

    def loop(self):
        while(True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if ret:
                img = cv2.resize(frame,(self.input_shape[1], self.input_shape[0]))
                # Operations on the frame
                img = np.array(img) / 255.0
                pred = self.model.predict(img.reshape((1, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
                # draw results on image
                for n, rects in enumerate(pred[0]):
                    for r in range(self.nb_boxes):
                        row = int(n / self.grid_h)
                        col = n - (row * self.grid_w)
                        label = rects[r]
                        cls = np.argmax(label[:self.class_nb])
                        if cls:
                            x = (label[self.class_nb + 0] * self.cell_w) + (col * self.cell_w)
                            y = (label[self.class_nb + 1] * self.cell_h) + (row * self.cell_h)
                            w = label[self.class_nb + 2] * self.img_w
                            h = label[self.class_nb + 3] * self.img_h
                            pt1 = (int(x - (w / 2)), int(y - (h / 2)))
                            pt2 = (int(x + (w / 2)), int(y + (h / 2)))
                            cv2.rectangle(img, pt1, pt2, colors[cls])
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
@click.option("-c", "class_nb", default=1, help="number of classes")
@click.option("-b", "nb_boxes", default=1, help="number of boxes per cell")
def main(model, class_nb, nb_boxes, rec, video_file):
    tracker = Tracker(model, class_nb, nb_boxes, rec, video_file)
    tracker.loop()
    tracker.stop()

if __name__ == "__main__":
    main()
