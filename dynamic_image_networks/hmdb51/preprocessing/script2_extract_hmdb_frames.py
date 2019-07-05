import cv2
from pathlib import Path
from dynamicimage import get_video_frames


def main():
    data_path = Path(r'E:\hmdb51_org\data',)
    out_path = Path(r'E:\hmdb51_org\frames')
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
        video_paths = subfolder.glob('*.avi')
        for video_path in video_paths:
            # Create an output folder for that video's frames.
            out_frame_folder = out_category_subfolder / video_path.stem
            out_frame_folder.mkdir()

            # Save the frames of the video. This process could be accelerated greatly by using ffmpeg if
            # available.
            # cmd = f'ffmpeg -i "{video_path}" -vf fps={fps} -q:v 2 -s {target_resolution[1]}x{target_resolution[0]} "{output_dir / "%06d.jpg"}"'
            frames = get_video_frames(str(video_path))
            for i, frame in enumerate(frames):
                frame = cv2.resize(frame, (224, 224))
                cv2.imwrite(str(out_frame_folder / (str(i).zfill(6) + '.jpg')), frame)


if __name__ == '__main__':
    main()
