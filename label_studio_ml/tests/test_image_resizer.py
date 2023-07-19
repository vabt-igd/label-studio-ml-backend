import os
import string
from random import SystemRandom
from urllib import request

import cv2
import numpy as np
from PIL import Image
from label_studio_converter.brush import encode_rle
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm


def rlencode(x, dropna=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.

    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.

    Returns
    -------
    start positions, run lengths, run values

    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]

    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]

    return starts, lengths, values

url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
request.urlretrieve(url, 'sam_vit_b_01ec64.pth')
sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
sam.to(device='cpu')
model = SamAutomaticMaskGenerator(sam, output_mode='binary_mask')


TARGET_IMAGE_PXL_COUNT = 1920*1080
image_path = "BOX0001_23_02_23__10_53_22.jpg"
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
cv2.imshow('Input Image', image)
cv2.waitKey(0)
h, w = image.shape[:2]
pixel_count = h * w

if pixel_count > TARGET_IMAGE_PXL_COUNT:
    scale_factor = TARGET_IMAGE_PXL_COUNT / pixel_count
    max_width = w * scale_factor
    max_height = h * scale_factor
    ratio_gcd = np.gcd(w, h)
    new_width = w / ratio_gcd
    new_height = h / ratio_gcd
    ratio_width = np.floor(max_width / new_width) * new_width
    ratio_height = np.floor(max_height / new_height) * new_height
    scaled_size = (max_width, max_height)

    if ratio_gcd > 1:
        scaled_size = (ratio_width, ratio_height)

    image = cv2.resize(image, np.array(scaled_size).astype(int), 0, 0, interpolation=cv2.INTER_AREA)

cv2.imshow('Downscaled Image', image)
cv2.waitKey(0)

# generate masks
all_scores = []
results = []
masks = model.generate(image)
for mask in tqdm(masks):
    segmentation = mask.get('segmentation')
    score = mask.get('stability_score')
    mask_id = ''.join(SystemRandom().choice(string.ascii_uppercase
                                            + string.ascii_lowercase
                                            + string.digits)
                      for _ in
                      range(10))

    all_scores.append(score)
    print(f'{os.linesep}Current mask:{os.linesep}'
          f' - bounding box:\t{mask.get("bbox")}{os.linesep}'
          f' - score:\t\t{score}{os.linesep}'
          f' - random id:\t\t{mask_id}{os.linesep}')

    _result_mask = np.zeros(image.shape[:2], dtype=np.uint16)  # convert result mask to mask
    result_mask = _result_mask.copy()
    result_mask[segmentation > 0] = 255
    result_mask = result_mask.astype(np.uint8)

    # convert mask to RGBA image
    rgbimg = cv2.merge((result_mask.copy(), result_mask.copy(), result_mask.copy(), result_mask.copy()))
    print(f'{os.linesep} -> SAM result mask successfully converted to an image!{os.linesep}')
    cv2.imshow('SAM Mask', rgbimg)
    cv2.waitKey(0)

    # upscale image if necessary
    if pixel_count > TARGET_IMAGE_PXL_COUNT:
        rgbimg = cv2.resize(rgbimg, np.array((w, h)).astype(int), 0, 0, interpolation=cv2.INTER_CUBIC)
    print(f'{os.linesep} -> mask successfully upscaled!{os.linesep}')

    # get pixels from image
    # pix = np.array(rgbimg)
    # encode to rle
    result_mask = encode_rle(rgbimg.flatten())

    # for each task, return classification results in the form of "choices" pre-annotations
    results.append(
        {
            'type': 'brushlabels',
            # 'id': mask_id,
            'value': {
                'format': 'rle',
                'rle': result_mask,
                'brushlabels': ['segment_' + mask_id]
            },
            'score': score
        },
    )

avg_score = sum(all_scores) / max(len(all_scores), 1)

# print(f'Segmentation results:{os.linesep}{results}{os.linesep}')
print(f'Segmentation finished! Sending results to server...')

if pixel_count > TARGET_IMAGE_PXL_COUNT:
    rgbimg = cv2.resize(image, np.array((w, h)).astype(int), 0, 0, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Upscaled Image', rgbimg)
    cv2.waitKey(0)