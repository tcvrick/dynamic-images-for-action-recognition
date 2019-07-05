import cv2
import numpy as np
from pathlib import Path
from dynamicimage import get_dynamic_image


WINDOW_LENGTH = 10
STRIDE = 6


def main():
    data_path = Path(r'E:\hmdb51_org\frames',)
    out_path = Path(r'E:\hmdb51_org\multiple_dynamic_images')
    out_path.mkdir()

    # Locate folder with the HMDB51 data.
    data_path = Path(data_path)
    print(f'Loading HMDB51 data from [{data_path.resolve()}]...')

    # Iterate over each category (sub-folder).
    categories = list(data_path.glob('*/'))

    for subfolder in categories:
        # Make output sub-folder for each category.
        out_category_subfolder = out_path / subfolder.stem
        out_category_subfolder.mkdir()

        # Iterate over each video in the category and extract the frames.
        frame_folder_paths = subfolder.glob('*/')
        for frame_folder in frame_folder_paths:
            # Create an output folder for that video's frames.
            out_frame_folder = out_category_subfolder / frame_folder.stem
            out_frame_folder.mkdir()

            frames = np.array([cv2.imread(str(x)) for x in frame_folder.glob('*.jpg')])
            for i in range(0, len(frames) - WINDOW_LENGTH, STRIDE):
                chunk = frames[i:i + WINDOW_LENGTH]
                assert len(chunk) == WINDOW_LENGTH

                dynamic_image = get_dynamic_image(chunk)
                cv2.imwrite(str(out_frame_folder / (str(i).zfill(6) + '.jpg')), dynamic_image)


if __name__ == '__main__':
    main()
