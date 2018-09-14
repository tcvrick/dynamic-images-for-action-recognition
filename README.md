# Python Dynamic Images for Action Recognition

Python implementation of the dynamic image technology discussed in 'Dynamic Image Networks for Action Recognition' by Bilen et al.
Their paper and GitHub can be found as follows:
* https://ieeexplore.ieee.org/document/7780700/
* https://github.com/hbilen/dynamic-image-nets

If you are planning on using this, please verify the correctness of the implementation for your provided inputs and outputs.

## Installation

Clone the directory, and install the requirements specified in the "requirements.txt" file.
~~~~
pip install -r requirements.txt
~~~~

## Example Usage
~~~~
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
~~~~

## Example Output
Source Video: https://www.youtube.com/watch?v=fXMDubfvoQE

![Dynamic Image Example](dynamic_image_example.JPG)