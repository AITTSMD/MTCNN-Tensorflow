#coding:utf-8
import os
import numpy as np


class IMDB(object):
    def __init__(self, name, image_set, root_path, dataset_path, mode='train'):
        self.name = name + '_' + image_set
        print(self.name)
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path
        self.mode = mode

        self.classes = ['__background__', 'face']
        self.num_classes = 2
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)

    @property
    def cache_path(self):
        """Make a directory to store all caches

        Parameters:
        ----------
        Returns:
        -------
        cache_path: str
            directory to store caches
        """
        cache_path = os.path.join(self.root_path, 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def load_image_set_index(self):
        """Get image index

        Parameters:
        ----------
        Returns:
        -------
        image_set_index: str
            relative path of image
        """
        image_set_index_file = os.path.join(self.data_path, 'imglists', self.image_set + '.txt')
        print("image_set_index_file", image_set_index_file)
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_set_index

    def gt_imdb(self):
        """Get and save ground truth image database

        Parameters:
        ----------
        Returns:
        -------
        gt_imdb: dict
            image database with annotations
        """
        # cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        #    with open(cache_file, 'rb') as f:
        #        imdb = cPickle.load(f)
        #    print '{} gt imdb loaded from {}'.format(self.name, cache_file)
        #    return imdb
        gt_imdb = self.load_annotations()
        # with open(cache_file, 'wb') as f:
        #    cPickle.dump(gt_imdb, f, cPickle.HIGHEST_PROTOCOL)
        return gt_imdb

    def image_path_from_index(self, index):
        """Given image index, return full path

        Parameters:
        ----------
        index: str
            relative path of image
        Returns:
        -------
        image_file: str
            full path of image
        """
        if not os.path.exists(index):
            image_file = os.path.join(self.data_path, index)
        else:
            image_file = index
        if not image_file.endswith('.jpg'):
            image_file = image_file + '.jpg'
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def load_annotations(self):
        """Load annotations

        Parameters:
        ----------
        Returns:
        -------
        imdb: dict
            image database with annotations
        """
        annotation_file = os.path.join(self.data_path, 'imglists', self.image_set + '.txt')
        print(os.path.join(self.data_path, 'imglists', self.image_set + '.txt'))
        assert os.path.exists(annotation_file), 'annotations not found at {}'.format(annotation_file)
        print("开始读取annotation文件", annotation_file)
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()

        imdb = []
        print(self.num_images)
        for i in range(self.num_images):
            annotation = annotations[i].strip().split(' ')
            index = annotation[0]
            if (int(i) + 1) % 10000 == 0:
                print("index:", index)
            # print(index)
            im_path = self.image_path_from_index(index)
            # print("im_path", im_path)
            imdb_ = dict()
            imdb_['image'] = im_path
            if self.mode == 'test':
                #                gt_boxes = map(float, annotation[1:])
                #                boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
                #                imdb_['gt_boxes'] = boxes
                pass
            else:
                label = annotation[1]
                imdb_['label'] = int(label)
                imdb_['flipped'] = False
                imdb_['bbox_target'] = np.zeros((4,))
                if len(annotation[2:]) == 4:
                    bbox_target = annotation[2:]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)

            imdb.append(imdb_)
        return imdb

    def append_flipped_images(self, imdb):
        """append flipped images to imdb

        Parameters:
        ----------
        imdb: imdb
            image database
        Returns:
        -------
        imdb: dict
            image database with flipped image annotations added
        """
        print('append flipped images to imdb', len(imdb))
        for i in range(len(imdb)):
            imdb_ = imdb[i]
            m_bbox = imdb_['bbox_target'].copy()
            m_bbox[0], m_bbox[2] = -m_bbox[2], -m_bbox[0]

            entry = {'image': imdb_['image'],
                     'label': imdb_['label'],
                     'bbox_target': m_bbox,
                     'flipped': True}

            imdb.append(entry)
        self.image_set_index *= 2
        return imdb

    def write_results(self, all_boxes):
        """write results

        Parameters:
        ----------
        all_boxes: list of numpy.ndarray
            detection results
        Returns:
        -------
        """
        print('Writing fddb results')
        # res_folder = os.path.join(self.cache_path, 'results')
        res_folder = os.path.join('./FDDB', 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        # save results to fddb format
        filename = os.path.join(res_folder, self.image_set + '-out.txt')
        with open(filename, 'w') as f:
            for im_ind, index in enumerate(self.image_set_index):
                # print("index: ", index)
                # f.write('%s\n' % index.replace('/world/data-c7/zhangboyu/data/FDDB/', '')[:-4])
                f.write('%s\n' % index)
                # print("index.replace[:-4]: ", index.replace('/world/data-c7/zhangboyu/data/FDDB/', '')[:-4])
                dets = all_boxes[im_ind]
                f.write('%d\n' % dets.shape[0])
                if len(dets) == 0:
                    continue
                for k in range(dets.shape[0]):
                    f.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.5f}\n'.
                            format(dets[k, 0], dets[k, 1], dets[k, 2] - dets[k, 0], dets[k, 3] - dets[k, 1],
                                   dets[k, 4]))
