# Python Dynamic Images for Action Recognition

Python implementation of the dynamic image technology discussed in 'Dynamic Image Networks for Action Recognition' by Bilen et al.
Their paper and GitHub can be found as follows:
* https://ieeexplore.ieee.org/document/7780700/
* https://github.com/hbilen/dynamic-image-nets

## Installation

Clone the directory, and install the module and it's pre-requisites by running:
~~~~
python setup.py install
~~~~

## Example Usage
~~~~
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
~~~~

## Example Output
Source Video: https://www.youtube.com/watch?v=fXMDubfvoQE

![Dynamic Image Example](dynamic_image_example.JPG)