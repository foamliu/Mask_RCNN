import cv2 as cv
import numpy as np
import os
import sys
import pickle
import pandas as pd

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def filter_by_class_id(r, target_id):
    print('class_ids.shape: ' + str(r['class_ids'].shape))
    print('rois.shape: ' + str(r['rois'].shape))
    print('masks.shape: ' + str(r['masks'].shape))
    print('scores.shape: ' + str(r['scores'].shape))
    class_ids = np.zeros((1,), np.int32)
    rois = np.zeros((1, 4))
    masks = np.zeros((480, 640, 1))
    scores = np.zeros((1,))
    j = 0
    for i in range(len(r['class_ids'])):
        if r['class_ids'][i] == target_id:
            class_ids[j] = r['class_ids'][i]
            rois[j] = r['rois'][i]
            masks[:, :, j] = r['masks'][:, :, i]
            scores[j] = r['scores'][i]
            j += 1

    return class_ids, rois, masks, scores


class App:
    def __init__(self):
        pass

    def run(self):
        pkl_file = open('data.pkl', 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()

        cap = cv.VideoCapture('joined_files.ts')
        fourcc = cv.VideoWriter_fourcc(*'MPEG')
        out = cv.VideoWriter('output.avi', fourcc, 10.0, (640, 480))

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            image = frame
            if not ret:
                print('ret is False')
                break

            r = data[frame_idx]
            class_ids = pd.Series(r['class_ids'])
            num_persons = 0

            if len(class_ids) > 0:
                class_ids, rois, masks, scores = filter_by_class_id(r, 1)
                image = visualize.display_instances(image, rois, masks, class_ids, class_names, scores)
                num_persons = len(class_ids)

            draw_str(image, (20, 20), '%d person(s) detected, frame_idx: %d.' % (num_persons, frame_idx))
            frame_idx = frame_idx + 1

            cv.imshow('image', image)
            out.write(image)

            ch = cv.waitKey(1)
            if ch == 27:
                break

        cap.release()
        out.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    App().run()
