import os
import string
from random import SystemRandom
from typing import List, Dict, Any
from urllib import request
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
import cv2
import numpy as np
from PIL import Image

from label_studio_converter.brush import encode_rle
from label_studio_tools.core.utils.io import get_data_dir
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, logger

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
request.urlretrieve(url, "sam_vit_b_01ec64.pth")


class SAMBackend(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(SAMBackend, self).__init__(**kwargs)

        # default Label Studio image upload folder
        self.image_dir = os.path.join(get_data_dir(), 'media', 'upload')
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')

        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        device = "cpu"  # "cuda"
        logger.debug(f'Model config:{os.linesep}'
                     f' - checkpoint:\t{sam_checkpoint}{os.linesep}'
                     f' - model type:\t{model_type}{os.linesep}'
                     f' - device used:\t{device}{os.linesep}')

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

    def predict(self, tasks, **kwargs):
        results = []
        all_scores = []
        # Get annotation tag first, and extract from_name/to_name keys from the labeling config to make predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]

        logger.debug(f'Task to complete: {len(tasks)}')
        for task in tasks:
            logger.debug(f'Current task: {task}')
            labels = []
            image_url = self._get_image_url(task)
            image_path = get_image_local_path(image_url, image_dir=self.image_dir)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            _result_mask = np.zeros(image.shape[:2], dtype=np.uint16)

            # loading context
            context = kwargs.get('context')
            clicks = context.get('result', [])
            for cl in clicks:
                labels.extend(cl.get('brushlabels', []))

            h, w, c = image.shape

            # generate masks
            masks: List[Dict[str, Any]] = self.sam_mask_generator.generate(image)
            for mask in masks:
                segmentation = mask.get('segmentation')
                score = mask.get('stability_score')
                all_scores.append(score)

                _result_mask = np.zeros(image.shape[:2], dtype=np.uint16)  # convert result mask to mask
                result_mask = _result_mask.copy()
                result_mask[segmentation > 0.5] = 255
                result_mask = result_mask.astype(np.uint8)
                # convert mask to RGBA image
                got_image = Image.fromarray(result_mask)
                rgbimg = Image.new("RGBA", got_image.size)
                rgbimg.paste(got_image)

                datas = rgbimg.getdata()
                # make pixels transparent
                new_data = []
                for item in datas:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        new_data.append((0, 0, 0, 0))
                    else:
                        new_data.append(item)
                rgbimg.putdata(new_data)
                # get pixels from image
                pix = np.array(rgbimg)
                # rgbimg.save("test.png")
                # encode to rle
                result_mask = encode_rle(pix.flatten())

                # for each task, return classification results in the form of "choices" pre-annotations
                results.append(
                    [
                        {
                            'original_width': w,
                            'original_height': h,
                            'value': {
                                'format': 'rle',
                                'rle': result_mask,
                                'brushlabels': labels
                            },
                            'id': ''.join(
                                SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                                for _ in
                                range(10)),
                            'from_name': from_name,
                            'to_name': to_name,
                            'type': 'brushlabels',
                            'score': score
                        },
                    ]
                )

        avg_score = sum(all_scores) / max(len(all_scores), 1)

        logger.debug(f"Segmentation results:{os.linesep}{results}")

        return [{
            'result': results,
            'score': avg_score
        }]

    def _get_image_url(self, task):
        data_values = list(task['data'].values())
        image_url = data_values[0]
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
                logger.warning(f'Can\'t generate resigned URL for {image_url}. Reason: {exc}')
        return image_url
