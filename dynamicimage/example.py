import glob
import cv2
from dynamicimage import get_dynamic_image


def main():
    frames = glob.glob('./example_frames/*.jpg')
    frames = [cv2.imread(f) for f in frames]

    dyn_image = get_dynamic_image(frames, normalized=True)
    cv2.imshow('', dyn_image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
