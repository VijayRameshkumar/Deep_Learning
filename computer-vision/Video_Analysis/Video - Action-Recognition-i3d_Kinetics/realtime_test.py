import cv2
from src import preprocess
import argparse
from i3d_inception import Inception_Inflated3d

# INPUT_SHAPE = 55
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400

LABEL_MAP_PATH = 'data/label_map.txt'
# load the kinetics classes
kinetics_classes = [x.strip() for x in open(LABEL_MAP_PATH, 'r')]

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", help="input video Path", type=str)
args = parser.parse_args()

video = cv2.VideoCapture(args.video_path)

ret,frame = video.read()

while(1):
    ret ,frame = cap.read()

    if ret == True:
        img = preprocess.pre_process_rgb(img)
        new_img = np.reshape(img, (1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
        rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_kinetics_only',
                input_shape=(INPUT_SHAPE, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
        rgb_logits = rgb_model.predict(rgb_sample)

            # produce softmax output from model logit for class probabilities
        sample_logits = rgb_logits[0] # we are dealing with just one example
        sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))
        sorted_indices = np.argsort(sample_predictions)[::-1]
        print('\nNorm of logits: %f' % np.linalg.norm(sample_logits))
        for index in sorted_indices[0]:
            print(sample_predictions[index], sample_logits[index], kinetics_classes[index])
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()