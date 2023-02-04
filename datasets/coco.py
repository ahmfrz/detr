# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as CT
import numpy as np
from PIL import Image
import cv2


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=512),
                T.Compose([
                    T.RandomResize([400, 500]),
                    T.RandomSizeCrop(384, 480),
                    T.RandomResize(scales, max_size=512),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([400], max_size=512),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    
    if args.has_custom_detection_class:
        dataset = CocoAugmented(
            img_folder, ann_file, transforms=get_transform(args.transform_type, image_set), return_masks=args.masks)
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset

def get_transform(transform_type, image_set):
    scales = [480, 512]

    if image_set == 'val':
        transforms = A.Compose([
            A.ToGray(always_apply=True),
            A.RandomSizedBBoxSafeCrop(scales[0], scales[1])
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    else:
        if transform_type == 'geometric':
            transforms = A.Compose([
                A.ToGray(always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent=np.random.random_sample(),p=0.5),
                A.Affine(rotate=np.random.randint(1,359), p=0.5),
                A.RandomSizedBBoxSafeCrop(350, 350)
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        elif transform_type == 'randomerasing':
            transforms = CT.Compose([
                CT.PILToTensor(),
                CT.Grayscale(3),
                CT.RandomErasing(p=0.5, value='random'),
                CT.ConvertImageDtype(torch.float32),
                CT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif transform_type == 'noise':
            transforms = A.Compose([
                A.ToGray(always_apply=True),
                A.GaussNoise(p=0.5),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        elif transform_type == 'copypaste':
            transforms = A.Compose([
                CopyPaste(num_of_copies=np.random.randint(0, 5))
                ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        elif transform_type == 'geometric+noise':
            transforms = A.Compose([
                A.ToGray(always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.5),
                A.Affine(translate_percent=np.random.random_sample(),p=0.5),
                A.Affine(rotate=np.random.randint(1,359), p=0.5),
                A.RandomSizedBBoxSafeCrop(350, 350), 
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        else:
            raise ValueError(f'Invalid transform type: {transform_type}')

    return transforms

class CopyPaste(A.DualTransform):
    def __init__(self, num_of_copies=2, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self._num_of_copies = num_of_copies
    
    def apply(self, image, anns, coco, **params) -> np.ndarray:
        return self.copypaste_transform(self._num_of_copies, image, anns, coco)

    def apply_to_bbox(self, bbox, **params):
        return bbox
    
    def get_params_dependent_on_targets(self, params):
        return {'anns': params['anns'], 'coco': params['coco']}
    
    @property
    def targets_as_params(self):
        return ['image','bboxes', 'anns', 'coco']

    @classmethod
    def copypaste_transform(cls, num_of_copies, image, anns, coco, **kwargs):
        objects = cls.extract_objects(image, anns, coco)
        img_obj, ann_obj = objects[0]
        mask_obj = coco.annToMask(ann_obj)
        max_tries = 10

        for i in range(num_of_copies):
            # check if x,y overlap with other objects
            mask =np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for ann in anns:
                mask = np.maximum(coco.annToMask(ann), mask)
            
            trial_count = 0
            has_overlap = True
            while has_overlap and trial_count < max_tries:
                trial_count+=1
                # generate random positions
                x, y = np.random.randint(
                    0, image.shape[1]), np.random.randint(0, image.shape[0])

                mask_bg_cropped = mask[y:y+img_obj.shape[0], x:x+img_obj.shape[1]]
                mask_obj_cropped = cv2.resize(
                    mask_obj, (mask_bg_cropped.shape[1], mask_bg_cropped.shape[0]))

                has_overlap = np.bitwise_and(mask_bg_cropped, mask_obj_cropped).any()
                
                has_overlap = has_overlap or (
                    (y+img_obj.shape[0]) > image.shape[0]) or ((x+img_obj.shape[1]) > image.shape[1])
            
            if trial_count < max_tries:
                # blend image
                alpha = np.ones(img_obj.shape[:2], dtype=np.float32) * 0.7
                alpha = np.dstack((alpha, alpha, alpha))
                img_obj_alpha = np.concatenate((img_obj, alpha), axis=2)
                
                # blend with object
                image[y:y+img_obj.shape[0], x:x+img_obj.shape[1],
                    :] = image[y:y+img_obj.shape[0], x:x+img_obj.shape[1], :] * (1-alpha) + img_obj_alpha[:,:,:3] * alpha
            
        return image

    @staticmethod
    def extract_objects(img, anns, coco):
        outputs = []
        for ann in anns:
            mask = coco.annToMask(ann)
            img_cropped = img * mask[:,:,np.newaxis]
            rows, cols = np.where(mask)
            top_row, bottom_row = rows.min(), rows.max()
            left_col, right_col = cols.min(), cols.max()
            
            if top_row == bottom_row:
                bottom_row += 1
            
            if left_col == right_col:
                right_col += 1
            
            outputs.append((img_cropped[top_row:bottom_row, left_col:right_col], ann))
        return outputs

class CocoAugmented(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoAugmented, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target_org = super(CocoAugmented, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target_org}
        img, target = self.prepare(img, target)
        bboxes, cats = self.get_bboxes_and_cats_from_anns(target_org)
        if self._transforms is not None:
            if not isinstance(self._transforms, CT.Compose):
                transformed = self._transforms(
                    image=np.array(img), bboxes=bboxes, category_ids=cats, coco=self.coco, anns=target_org)

                img = transformed['image']
                target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32).reshape(-1, 4)
                target['labels'] = torch.tensor(transformed['category_ids'], dtype=torch.int64)
                w, h = img.shape[1], img.shape[0]
                target['size'] = torch.as_tensor([int(h), int(w)])
                
                # normalize image and convert to tensor
                normalize = A.Compose([
                    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ToTensorV2()
                ],bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
                img = normalize(
                    image=transformed['image'], bboxes=transformed['bboxes'], category_ids=transformed['category_ids'])['image']
            else:
                # pytorch transform
                img = self._transforms(img)

        return img, target

    @staticmethod
    def get_bboxes_and_cats_from_anns(anns):
        return [ann['bbox'] for ann in anns], [ann['category_id'] for ann in anns]
