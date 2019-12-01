import click
import cv2
import os
import json
import ctypes
from const import colors, img_size

class Mask(object):

    def __init__(self, images_path, labels_file):
        self.images_path = images_path
        self.labels_file = labels_file
        self.key = None
        self.pts = []
        self.drag = False
        self.class_label = 0
        self.images = [f if f.endswith(".png") or f.endswith(".jpg") else 0 for f in os.listdir(self.images_path)]
        self.labels = {}
        cv2.namedWindow("Frame")
        if os.path.exists(self.labels_file):
            with open(self.labels_file, mode="r") as f:
                self.labels = json.load(f)
        print("Draw masks : Lclick to start area, Rclick to set points\n(last point automatically added)\n[0-9] : select current class\nN : next img \nP : previous img \nD : delete img \nA : clear label\n", flush=True)

    def append_label(self, img_name, pts):
        if img_name in self.labels:
            if str(self.class_label) in self.labels[img_name]:
                self.labels[img_name][str(self.class_label)].append(pts)
            else:
                self.labels[img_name][str(self.class_label)] = [pts]
        else:
            self.labels[img_name] = {}
            self.labels[img_name][str(self.class_label)] = [pts]

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
            self.pts.append((x, y))
        elif self.drag == True and event == cv2.EVENT_RBUTTONDOWN:
            self.pts.append((x, y))
            cv2.line(param[0], self.pts[-2], self.pts[-1], color)
        elif self.drag == True and event == cv2.EVENT_MOUSEMOVE:
            cv2.line(tmp_img, self.pts[-1], (x, y), color)
        elif self.drag == True and event == cv2.EVENT_LBUTTONUP:
            self.drag = False
            cv2.line(param[0], self.pts[-1], self.pts[0], color)
            self.append_label(param[1], [(pt[0] / tmp_img.shape[1], pt[1] / tmp_img.shape[0]) for pt in self.pts])
            self.pts = []
        cv2.imshow('Frame',tmp_img)

    def loop(self):
        i = 0
        while True:
            if i >= 0 and i < len(self.images) and self.images[i] != None:
                # open image and draw ROI
                print(self.images_path, self.images[i])
                img = cv2.imread(os.path.join(self.images_path, self.images[i]))
                img = cv2.resize(img, img_size)
                if self.images[i] in self.labels:
                    for class_label in self.labels[self.images[i]]:
                        color = colors[int(class_label)]
                        for mask in self.labels[self.images[i]][class_label]:
                            for j, _ in enumerate(mask):
                                if j > 0:
                                    pt1 = (int(mask[j-1][0] * img.shape[1]), int(mask[j-1][1] * img.shape[0]))
                                    pt2 = (int(mask[j][0] * img.shape[1]), int(mask[j][1] * img.shape[0]))
                                    cv2.line(img, pt1, pt2, color)
                            pt1 = (int(mask[j][0] * img.shape[1]), int(mask[j][1] * img.shape[0]))
                            pt2 = (int(mask[0][0] * img.shape[1]), int(mask[0][1] * img.shape[0]))
                            cv2.line(img, pt1, pt2, color)
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
                elif self.key == ord('a') and i <= len(self.images) - 1:
                    self.clean_label(self.images[i])
                elif self.key == ord('n') and i < len(self.images) - 1:
                    i += 1
                elif self.key == ord('p') and i > 0:
                    i -= 1
                elif self.key == ord('d'):
                    # Remove file
                    if ctypes.windll.user32.MessageBoxW(0, "This will remove file !", "Warning", 1) == 1:
                        os.remove(self.images[i])
                        self.remove_label(self.images[i])
                        self.images[i] = None
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
@click.option('-o', 'labels_file', default="data/masks.json", help="path to store labels")
def main(images_path, labels_file):
    mask = Mask(images_path, labels_file)
    mask.loop()

if __name__ == "__main__":
    main()
