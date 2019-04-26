# WFLW layout
# WFLW
# ├── WFLW_annotations
# │   ├── list_98pt_rect_attr_train_test
# │   └── list_98pt_test
# └── WFLW_images
#     ├── 0--Parade
#     ├── 10--People_Marching
#     ├── 11--Meeting
#     ├── ....
#
#

import os
import os.path as osp

import cv2
import numpy as np

import torch

class WFLW:
    def __init__(self, path:str):
        # setup paths
        self.root = path
        self.train_path = self.setPath('WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt')
        self.valid_path = self.setPath('WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt')
        self.test_directory = self.setPath('WFLW_annotations/list_98pt_test/')
        self.images_directory = self.setPath('WFLW_images/')
        # parse annotations
        self.train_annotations = self.parse_annotations(self.train_path, True)
        self.valid_annotations = self.parse_annotations(self.valid_path, True)


    def setPath(self, subpath):
        return osp.join(self.root, subpath)

    def parse_annotations(self, annotations_filepath:str, training:bool):
        with open(annotations_filepath, encoding='utf-8') as f:
            annlist = f.readlines()

        annlist = [ann.rstrip().split() for ann in annlist]
        landmarks = []
        rects = []
        attrs = []
        paths = []

        if training:
            # x0 y0  ...  x97 y97  x_min_rect y_min_rect x_max_rect y_max_rect attrs path
            for ann in annlist:
                landmarks.append(np.fromiter(ann[:196], dtype=np.float32))
                rects.append([int(a) for a in ann[196:200]])
                attrs.append([int(a) for a in ann[200:206]])
                paths.append(ann[206])

        else:
            for ann in annlist:
                landmarks.append(ann[:196])
                paths.append(ann[196])
        return {'landmarks': landmarks,
                'rects': rects,
                'attrs': attrs,
                'paths': paths}


    def _data_generator(self, batch_size, mode):
        """
        mode: only "train","valid","test" allowed
        """

        if mode == 'train':
            data = self.train_annotations
        elif mode == 'valid':
            data = self.valid_annotations
        elif mode == 'test':
            pass
        else:
            raise KeyError(f"mode {mode} invalid, Only `train`, `valid`, 'test' allowed")

        data_size = len(data['landmarks'])
        num_batches = data_size // batch_size + 1

        for batch_idx in range(num_batches):
            start_index = batch_idx * batch_size
            end_index = min((batch_idx + 1) * batch_size, data_size)
            landmarks = data['landmarks'][start_index:end_index]
            rects = data['rects'][start_index:end_index]
            attrs = data['attrs'][start_index:end_index]
            paths = data['paths'][start_index:end_index]
            yield landmarks, rects, attrs, paths

    def data_generator(self, batch_size, mode):
        for landmarks, rects, attrs, paths in self._data_generator(batch_size, mode):
            landmarklist = []
            facelist = []
            for landmark, face in map(self.data_prepare, landmarks, rects, attrs, paths):
                landmarklist.append(landmark)
                facelist.append(face)
            yield facelist, landmarklist


    def data_prepare(self, landmark, rect, attr, path):
        image = cv2.imread(osp.join(self.images_directory, path))
        # create new bbox
        margin = int((rect[2] - rect[0]) * 0.4)
        y_min = max(rect[1]-margin, 0)
        y_max = min(rect[3]+margin, image.shape[0])
        x_min = max(rect[0]-margin, 0)
        x_max = min(rect[2]+margin, image.shape[1])
        image = image[y_min:y_max, x_min:x_max, :]
        landmark[::2] -= x_min
        landmark[1::2] -= y_min
        return landmark, image


class Transformer(object):
    def __init__(self, target_height, target_width):
        self.height = target_height
        self.width = target_width

    def transform(self, image):
        return np.transpose(self.normalize(image), (2, 0, 1))

    def letterbox(self, image, landmark):
        """scale image to match input size of network.
        When training, padding offset and scale is not needed"""
        canvas = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        height, width = image.shape[:2]
        offset_x = 0
        offset_y = 0
        if height > width:
            scale = self.height / height
            width_ = int(width * scale)
            image_ = cv2.resize(image, (width_, self.height))
            offset_x = (self.width - width_) // 2
            canvas[:, offset_x:offset_x+width_, :] = image_[:, :, :]
        else:
            scale = self.width / width
            height_ = int(height * scale)
            image_ = cv2.resize(image, (self.width, height_))
            offset_y = (self.height - height_) // 2
            canvas[offset_y:offset_y+height_, :, :] = image_[:, :, :]
        # tranform landmark
        landmark *= scale
        landmark[::2] += offset_x
        landmark[1::2] += offset_y
        return canvas, landmark, (scale, offset_x, offset_y)

    def inverse_letterbox(self, landmark, scale, offset_x, offset_y):
        landmark[::2] -= offset_x
        landmark[1::2] -= offset_y
        landmark /= scale
        return landmark

    def normalize(self, image):
        return image / 255.0




if __name__ == '__main__':
    from vutils import plot
    dataset = WFLW('data/WFLW')
    ts = Transformer(target_height=224, target_width=224)
    for landmarks, images in dataset.data_generator(4, mode='valid'):
        for i, (landmark, image) in enumerate(zip(landmarks, images)):
            plot(image, landmark, "test/test_{}.png".format(i))
            new_image, new_landmark, _ = ts.letterbox(image, landmark)
            plot(new_image, new_landmark, "test/new_{}.png".format(i))
        break


