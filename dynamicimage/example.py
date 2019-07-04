import glob
import cv2
import dynamicimage
import numpy as np


def main():
    frames = glob.glob('./example_frames/*.jpg')
    frames = np.array([cv2.imread(f) for f in frames])

    dyn_image = dynamicimage.get_dynamic_image(frames, normalized=True)
    cv2.imshow('', dyn_image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
