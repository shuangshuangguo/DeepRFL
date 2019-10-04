import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def ratio(self):
        return float(self._data[2])

    @property
    def part_frames(self):
        return int(self._data[3])

    @property
    def label(self):
        return int(self._data[4])


class DataSet(data.Dataset):
    def __init__(self, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        #for flow, the function returns a list whose length is two
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, frames):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif frames > self.new_length:
            offsets = np.sort(randint(frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.part_frames > self.num_segments + self.new_length - 1:
            tick = (record.part_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        if record.part_frames <= self.num_segments + self.new_length - 1:
            offsets = np.zeros((self.num_segments,))
        else:
            tick = (record.part_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.test_mode and self.random_shift:
            segment_indices = self._sample_indices(record.num_frames)
            part_indices = self._sample_indices(record.part_frames)
            a, b, e = self.get(record, segment_indices)
            c, d, f = self.get_part(record, part_indices)
            return a, b, e, c, d, f
        if not self.test_mode and not self.random_shift:
            segment_indices = self._get_val_indices(record)
            return self.get_part(record, segment_indices)
        if self.test_mode:
            segment_indices = self._get_test_indices(record)
            return self.get_part(record, segment_indices)

    def get(self, record, indices):
        images = list()
        '''get the process_data and label and ratio of full_videos'''
        '''indices is the index of frame'''
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                # get the continuous images of frames due to the new_length
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
        # the length of list is 3 or 15
        # return the transformed images of frames
        process_data = self.transform(images)
        return process_data, record.label, record.ratio

    def get_part(self, record, indices):
        '''get the part videos data'''
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.part_frames:
                    p += 1
        # the length of list is 3 or 15
        process_data = self.transform(images)
        return process_data, record.label, record.ratio

    def __len__(self):
        return len(self.video_list)
