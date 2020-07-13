import glob
import cv2
from pathlib import Path
from dynamicimage import get_dynamic_image


def main():
    # Load the frames from the 'example_frames' folder and sort them numerically. This assumes that your frames
    # are enumerated as 0001.jpg, 0002.jpg, etc.
    frames = glob.glob('./example_frames/*.jpg')
    frames = sorted(frames, key=lambda x: int(Path(x).stem))
    frames = [cv2.imread(f) for f in frames]

    # Generate and display a normalized dynamic image.
    dyn_image = get_dynamic_image(frames, normalized=True)
    cv2.imshow('', dyn_image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
