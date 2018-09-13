import cv2
import numpy as np


""" 
Python implementation of technology discussed in 'Dynamic Image Networks for Action Recognition' by Bilen et al.
Their paper and GitHub can be found here: https://github.com/hbilen/dynamic-image-nets

If you're planning on using this, please verify the correctness of the implementation for your own inputs and outputs! 
"""


def get_dynamic_image(frames, normalized=True):
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""
    num_channels = frames[0].shape[2]
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image


def _get_channel_frames(iter_frames, num_channels):
    """ Takes a list of frames and returns a list of frame lists split by channel. """
    frames = [[] for channel in range(num_channels)]

    for frame in iter_frames:
        for channel_frames, channel in zip(frames, cv2.split(frame)):
            channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
    for i in range(len(frames)):
        frames[i] = np.array(frames[i])
    return frames


def _compute_dynamic_image(frames):
    """ Inspired by https://github.com/hbilen/dynamic-image-nets """
    num_frames, h, w, depth = frames.shape

    y = np.zeros((num_frames, h, w, depth))
    ids = np.ones(num_frames)

    fw = np.zeros(num_frames)
    for n in range(num_frames):
        cumulative_indices = np.array(range(n, num_frames)) + 1
        fw[n] = np.sum(((2*cumulative_indices) - num_frames) / cumulative_indices)

    for v in range(int(np.max(ids))):
        indv = np.array(np.where(ids == v+1))

        a1 = frames[indv, :, :, :]
        a2 = np.reshape(fw, (indv.shape[1], 1, 1, 1))
        a3 = a1 * a2
        y[v, :, :, :] = np.sum(a3[0], axis=0)

    return y[0]
