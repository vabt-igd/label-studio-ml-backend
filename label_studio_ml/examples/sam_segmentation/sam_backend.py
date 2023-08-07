import io
import json
import logging
import os
import string
from random import SystemRandom
from typing import Union
from urllib import request
from urllib.parse import urlparse

import boto3
import torch
from botocore.exceptions import ClientError
import cv2
import numpy as np

from label_studio_converter.brush import encode_rle
from label_studio_tools.core.utils.io import get_data_dir
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path

logger = logging.getLogger(__name__)

url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
request.urlretrieve(url, 'sam_vit_b_01ec64.pth')

DEBUG_IMAGE_PATH_SEG_OUT = os.path.join('images', 'in_raw')
DEBUG_IMAGE_PATH_IN = os.path.join('images', 'seg_results')
TARGET_IMAGE_PXL_COUNT = 2560*1440


class SAMBackend(LabelStudioMLBase):

    def __init__(self,
                 checkpoint_file: Union[str, None] = 'sam_vit_b_01ec64.pth',
                 image_dir: Union[str, None] = None,
                 score_threshold=0.5,
                 # device='cpu',  # 'cuda'
                 debug_segmentation_output=False,
                 **kwargs):
        """
        Load Segment Anything model for interactive segmentation from checkpoint.

        :param checkpoint_file: Absolute path to the SAM checkpoint
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs:
        """

        # don't forget to initialize base class...
        super(SAMBackend, self).__init__(**kwargs)

        self.checkpoint_file = checkpoint_file
        model_type = 'vit_b'
        self.score_thresh = score_threshold
        self.debug_segmentation_output = debug_segmentation_output

        if self.debug_segmentation_output:
            if not os.path.exists(DEBUG_IMAGE_PATH_SEG_OUT):
                os.makedirs(DEBUG_IMAGE_PATH_SEG_OUT)

            if not os.path.exists(DEBUG_IMAGE_PATH_IN):
                os.makedirs(DEBUG_IMAGE_PATH_IN)

        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            self.device = 'cuda:' + str(device_idx)
            self.device_str = 'cuda: ' + str(device_idx) + ' - ' + str(torch.cuda.get_device_name(device_idx))
            print(f'Inference using CUDA on device {device_idx}: {torch.cuda.get_device_name(device_idx)}{os.linesep}')
        else:
            self.device = 'cpu'
            from cpuinfo import get_cpu_info
            info = get_cpu_info()
            self.device_str = 'cpu: ' + info['brand_raw'] + " | Arch: " + info['arch_string_raw']
            print('Inference using the CPU')
            print(f'NOTE: This may be to slow for label studio to work reliably!{os.linesep}')

        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        print(f'{self.__class__.__name__} reads images from {self.image_dir}{os.linesep}')
        print(f'Model config:{os.linesep}'
              f' - checkpoint:\t{checkpoint_file}{os.linesep}'
              f' - model type:\t{model_type}{os.linesep}'
              f' - device used:\t{self.device_str}{os.linesep}')

        sam = sam_model_registry[model_type](checkpoint=checkpoint_file)
        sam.to(device=self.device)
        self.model = SamAutomaticMaskGenerator(sam, output_mode='binary_mask')

    def predict(self, tasks, **kwargs):
        results = []
        all_scores = []
        # Get annotation tag first, and extract from_name/to_name keys from the labeling config to make predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]

        print(f'Tasks to complete: {len(tasks)}{os.linesep}')
        for task in tqdm(tasks):
            print(f'{os.linesep}Current task: {task}{os.linesep}')
            image_url = self._get_image_url(task)
            image_path = get_image_local_path(image_url, image_dir=self.image_dir)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
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

            # generate masks
            all_scores = []
            results = []
            masks = self.model.generate(image)
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
                        'from_name': from_name,
                        'to_name': to_name,
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

        return [{
            'result': results,
            'score': avg_score
        }]

    def _get_image_url(self, task):
        image_url = list(task['data'].values())[0]
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate pre-signed URL for {image_url}. Reason: {exc}')
        return image_url


def _json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
