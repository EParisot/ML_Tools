import click
import cv2
import os
import json
import ctypes
from const import colors, img_size

class Bbox(object):

    def __init__(self, images_path, labels_file):
        self.key = None
        self.pt1 = None
        self.pt2 = None
        self.drag = False
        self.class_label = 0
        self.images_path = images_path
        self.images = []
        for f in os.listdir(self.images_path):
            if f.endswith(".jpg") or f.endswith(".png"):
                self.images.append(f)
        self.labels_file = labels_file
        self.labels = {}
        if os.path.exists(self.labels_file):
            with open(self.labels_file, mode="r") as f:
                self.labels = json.load(f)
        cv2.namedWindow("Frame")
        print("Draw ROI : Lclick to start rect and release to draw ROI\n[0-9] : select current class\nN : next \nP : previous \nD : delete \nA : absent\n", flush=True)

    def append_label(self, img, img_name):
        pt1 = [min(self.pt1[0], self.pt2[0]), min(self.pt1[1], self.pt2[1])]
        pt2 = [max(self.pt1[0], self.pt2[0]), max(self.pt1[1], self.pt2[1])]
        pt1[0] = round(pt1[0] / img.shape[1], 2)
        pt1[1] = round(pt1[1] / img.shape[0], 2)
        pt2[0] = round(pt2[0] / img.shape[1], 2)
        pt2[1] = round(pt2[1] / img.shape[0], 2)
        # transform [pt1, pt2] to [x, y, w, h]
        w = abs(pt2[0] - pt1[0])
        h = abs(pt2[1] - pt1[1])
        x = pt1[0] + w / 2
        y = pt1[1] + h / 2
        if img_name in self.labels:
            if str(self.class_label) in self.labels[img_name]:
                self.labels[img_name][str(self.class_label)].append([x, y, w, h])
            else:
                self.labels[img_name][str(self.class_label)] = [[x, y, w, h]]
        else:
            self.labels[img_name] = {}
            self.labels[img_name][str(self.class_label)] = [[x, y, w, h]]

    def clean_label(self, img_name):
        if img_name in self.labels and str(self.class_label) in self.labels[img_name]:
            self.labels[img_name][str(self.class_label)] = []

    def remove_label(self, img_name):
        if img_name in self.labels and str(self.class_label) in self.labels[img_name]:
            del self.labels[img_name][str(self.class_label)]
            self.labels = {}

    def mouse_drawing(self, event, x, y, flags, param):
        tmp_img = param[0].copy()
        color = colors[self.class_label]
        if self.drag == False and event == cv2.EVENT_LBUTTONDOWN:
            self.drag = True
            self.pt1 = (x, y)
        elif self.drag == True and event == cv2.EVENT_LBUTTONUP:
            self.drag = False
            if x > tmp_img.shape[1]:
                x = tmp_img.shape[1]
            elif x < 0:
                x = 0
            if y > tmp_img.shape[0]:
                y = tmp_img.shape[0]
            elif y < 0:
                y = 0
            self.pt2 = (x, y)
            cv2.rectangle(param[0], self.pt1, self.pt2, color)
            self.append_label(tmp_img, param[1])
        elif self.drag == True and event == cv2.EVENT_MOUSEMOVE:
            self.pt2 = (x, y)
            cv2.rectangle(tmp_img, self.pt1, self.pt2, color)
        cv2.imshow('Frame',tmp_img)

    def loop(self):
        i = 0
        while True:
            if i >= 0 and i < len(self.images) and self.images[i] != None:
                # open image and draw ROI
                img = cv2.imread(os.path.join(self.images_path, self.images[i]))
                img = cv2.resize(img, img_size)
                if self.images[i] in self.labels:
                    for class_label in self.labels[self.images[i]]:
                        color = colors[int(class_label)]
                        for rect in self.labels[self.images[i]][class_label]:
                            self.pt1 = (int((rect[0] - (rect[2] / 2)) * img.shape[1]), int((rect[1] - (rect[3] / 2)) * img.shape[0]))
                            self.pt2 = (int((rect[0] + (rect[2] / 2)) * img.shape[1]), int((rect[1] + (rect[3] / 2)) * img.shape[0]))
                            cv2.rectangle(img, self.pt1, self.pt2, color)
                cv2.imshow('Frame', img)
                cv2.setMouseCallback("Frame", self.mouse_drawing, param=[img, self.images[i]])
                # key events
                self.key = cv2.waitKey(0)
                if self.key == 27:
                    self.end()
                    return
                elif self.key == ord('à') or self.key == ord('0'):
                    self.class_label = 0
                elif self.key == ord('&') or self.key == ord('1'):
                    self.class_label = 1
                elif self.key == ord('é') or self.key == ord('2'):
                    self.class_label = 2
                elif self.key == ord('"') or self.key == ord('3'):
                    self.class_label = 3
                elif self.key == ord("'") or self.key == ord('4'):
                    self.class_label = 4
                elif self.key == ord('(') or self.key == ord('5'):
                    self.class_label = 5
                elif self.key == ord('-') or self.key == ord('6'):
                    self.class_label = 6
                elif self.key == ord('è') or self.key == ord('7'):
                    self.class_label = 7
                elif self.key == ord('_') or self.key == ord('8'):
                    self.class_label = 8
                elif self.key == ord('ç') or self.key == ord('9'):
                    self.class_label = 9
                elif self.key == ord('n') and i < len(self.images) - 1:
                    i += 1
                elif self.key == ord('p') and i > 0:
                    i -= 1
                elif self.key == ord('c') and i <= len(self.images) - 1:
                    self.clean_label(self.images[i])
                elif self.key == ord('d'):
                    # Remove file
                    if ctypes.windll.user32.MessageBoxW(0, "This will remove file !", "Warning", 1) == 1:
                        os.remove(self.images[i])
                        self.images[i] = None
                        self.remove_label(self.images[i])
                        i += 1
            elif i >= 0 and i < len(self.images) and self.images[i] == None:
                if self.key == ord('p'):
                    i -= 1
                else:
                    i += 1
            else:
                self.end()
                return
            
    def end(self):
        with open(self.labels_file, mode="w") as f:
            json.dump(self.labels, f)
        cv2.destroyAllWindows()

@click.command()
@click.option('-i', 'images_path', default="data", help="path to find images")
@click.option('-o', 'labels_file', default="bboxes.json", help="path to store labels")
def main(images_path, labels_file):
    bbox = Bbox(images_path, labels_file)
    bbox.loop()

if __name__ == "__main__":
    main()
