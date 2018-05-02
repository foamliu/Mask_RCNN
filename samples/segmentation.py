import cv2 as cv
import pickle

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

            masked_image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                       class_names, r['scores'])

            frame_idx = frame_idx + 1
            print(frame_idx)

            cv.imshow('image', masked_image)
            out.write(masked_image)

            ch = cv.waitKey(1)
            if ch == 27:
                break

        cap.release()
        out.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    App().run()
