import glob
import cv2
import numpy as np
import dynamicimage


def main():
    frames = glob.glob('./example_frames/*.jpg')
    frames = [cv2.imread(f) for f in frames]

    dyn_image = dynamicimage.get_dynamic_image(frames, normalized=True)
    cv2.imshow('', dyn_image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
