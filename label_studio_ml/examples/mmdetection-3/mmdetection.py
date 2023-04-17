import os
import logging
import string
from random import SystemRandom

import boto3
import io
import json

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
register_all_modules()


class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(self, config_file=None,
                 checkpoint_file=None,
                 image_dir=None,
                 labels_file=None, score_threshold=0.5, device='cpu', **kwargs):
        """
        Load MMDetection model from config and checkpoint into memory.
        (Check https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html#high-level-apis-for-testing-images)

        Optionally set mappings from COCO classes to target labels
        :param config_file: Absolute path to MMDetection config file (e.g. /home/user/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x.py)
        :param checkpoint_file: Absolute path MMDetection checkpoint file (e.g. /home/user/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs:
        """

        super(MMDetection, self).__init__(**kwargs)
        config_file = config_file or os.environ['config_file']
        checkpoint_file = checkpoint_file or os.environ['checkpoint_file']
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')

        print('Load new model from: ', config_file, checkpoint_file)
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.score_thresh = score_threshold

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

    def predict(self, tasks, **kwargs):
        all_scores = []
        results = []
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]
        classes = self.model.dataset_meta.get('classes')
        print(f'New prediction request!{os.linesep}')
        print(f'Model recognizes the following classes:{os.linesep}{classes}{os.linesep}')
        print(f'Tasks to complete: {len(tasks)}{os.linesep}')

        for task in tasks:
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            model_results = inference_detector(self.model, image_path).pred_instances
            img_width, img_height = get_image_size(image_path)
            print(f'Model predicted {len(model_results)} labels.{os.linesep}')

            for item in model_results:
                bboxes, label, scores = item['bboxes'], item['labels'], item['scores']
                output_label = classes[label]
                score = float(scores[-1])
                label_id = ''.join(SystemRandom().choice(string.ascii_uppercase
                                                         + string.ascii_lowercase
                                                         + string.digits))
                if score < self.score_thresh:
                    print(f'Prediction [{label}] : {output_label}{os.linesep} not accepted. '
                          f'Score too low: {score} < {self.score_thresh} (threshold)')
                    continue

                for bbox in bboxes:
                    bbox = list(bbox)
                    if not bbox:
                        continue

                    print(f'Prediction accepted:{os.linesep}'
                          f' - label:\t{output_label}{os.linesep}',
                          f' - bbox:\t{bbox}{os.linesep}',
                          f' - score:\t{score}{os.linesep}'
                          f' - random id:\t\t{label_id}{os.linesep}')

                    x, y, xmax, ymax = bbox[:4]
                    results.append(
                        {
                            'from_name': from_name,
                            'to_name': to_name,
                            'type': 'rectanglelabels',
                            'id': label_id,
                            'value': {
                                'rectanglelabels': [''.join(output_label)],
                                'x': float(x) / img_width * 100,
                                'y': float(y) / img_height * 100,
                                'width': (float(xmax) - float(x)) / img_width * 100,
                                'height': (float(ymax) - float(y)) / img_height * 100
                            },
                            'score': score
                        }
                    )
                    all_scores.append(score)

        avg_score = sum(all_scores) / max(len(all_scores), 1)
        # print(f'Classification results:{os.linesep}{results}{os.linesep}')
        print(f'Prediction finished! Sending results to server...')
        return [{
            'result': results,
            'score': avg_score
        }]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
