import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os, inspect
import glob
import cv2
from meaningful_splits import splits, ntu_splits
from kinetics_labels import get_kinetics_names

rng = np.random.RandomState(2017)

def np_del_by_val_1d(arr, *del_arrs):
    ret = np.array(arr)
    for del_arr in del_arrs:
        ret = np.delete(ret, [i for i in range(ret.shape[0]) if ret[i] in del_arr])
    return ret


def get_exp_classes(split_name, m=250, ntu=False):
    # NTU only removes non-normal classes from the 60, Kinetics filters by sorted success threshold
    if isinstance(split_name, (list, tuple, np.ndarray, np.generic)):
        normal_classes = list(split_name)
    else:
        if ntu:
            normal_classes = ntu_splits[split_name]
        else:
            normal_classes = splits[split_name]
    if ntu:
        abnormal_classes = np_del_by_val_1d(list(range(60)), normal_classes)
        return normal_classes, abnormal_classes
    # currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # parentdir = os.path.dirname(currentdir)
    # class_acc_path = os.path.join(parentdir, 'csv/class_accuracy_np.npy')
    class_acc_path = 'csv/class_accuracy_np.npy'
    class_acc_np = np.load(class_acc_path)
    class_num = class_acc_np.shape[0]
    unusable_classes = class_acc_np[m:, 0].astype(int)
    normal_classes = np_del_by_val_1d(normal_classes, unusable_classes)
    abnormal_classes = np_del_by_val_1d(np.arange(class_num), normal_classes, unusable_classes)
    return normal_classes, abnormal_classes

def np_load_frame(frame, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    # image_decoded = cv2.imread(filename)
    # image_resized = cv2.resize(frame, (resize_width, resize_height))
    # print("image resize start {}".format(frame.shape[0]))

    # image_resized = np.zeros_like(frame)
    image_resized = cv2.resize(frame, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(object):
    def __init__(self, video_folder, split, phase, inv_exp, ntu, resize_height=256, resize_width=256):
        self.dir = video_folder
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._split = split
        self._phase = phase

        self._inv = inv_exp
        self._ntu = ntu
        self.setup()

    def __call__(self, batch_size, time_steps, num_pred=1):
        video_info_list = list(self.videos.values())
        num_videos = len(video_info_list)

        print("number of videos {}".format(num_videos))
        clip_length = time_steps + num_pred
        resize_height, resize_width = self._resize_height, self._resize_width

        def train_video_clip_generator():
            v_id = -1
            while True:
                v_id = (v_id + 1) % num_videos

                video_info = video_info_list[v_id]

                cap = cv2.VideoCapture(video_info['path'])
                # vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                vid_length = video_info['length']
                # print("vid: {} - {} - {}".format(video_info['path'], vid_length, resize_width))
                if not vid_length:
                    continue

                # fps = int(cap.get(cv2.CAP_PROP_FPS))
                start = rng.randint(0, vid_length - clip_length)
                # start = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                # print("starting from frame: {}".format(start))
                video_clip = []
                for frame_id in range(start, start + clip_length):
                    ret, frame = cap.read()
                    if ret:
                        frame_resize = np_load_frame(frame, resize_height, resize_width)
                        # print("vid: {} read complete".format(video_info['path']))
                        video_clip.append(frame_resize)
                    else:
                        print('frame not loaded')
                video_clip = np.concatenate(video_clip, axis=2)
                yield video_clip

        def test_video_clip_generator():
            v_id = -1
            while v_id < num_videos-1:

                v_id = (v_id + 1)
                video_info = video_info_list[v_id]
                print("processing vid {}".format(video_info['path']))
                cap = cv2.VideoCapture(video_info['path'])
                vid_length = video_info['length']

                if not vid_length:
                    continue

                frame_id = -1
                video_clip_buffer = []

                while frame_id < vid_length-1:


                    # print("vid: {} - {} - {}".format(video_info['path'], vid_length, resize_width))
                    # while frame_id < clip_length:

                    ret, frame = cap.read()
                    # if ret:
                    frame_id += 1
                    video_clip_buffer.append(np_load_frame(frame, resize_height, resize_width))
                    # else:
                    #     video_clip_buffer.append(np.zeros((resize_height, resize_width, 3)))

                        # print("frame skipped")

                    if frame_id < clip_length-1:
                        continue

                    # print('frame id {}'.format(frame_id))
                    video_clip = np.concatenate(video_clip_buffer, axis=2)
                    video_clip_buffer.pop(0)
                    # cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    # print("starting from frame: {}".format(start))
                    # video_clip = []
                    # for frame_id in range(frame_id - clip_length, frame_id + 1):
                    #     ret, frame = cap.read()
                    #     if ret:
                    #         frame_resize = np_load_frame(frame, resize_height, resize_width)
                    #         # print("vid: {} read complete".format(video_info['path']))
                    #         video_clip.append(frame_resize)
                    #     else:
                    #         print('frame not loaded')
                    # video_clip = np.concatenate(video_clip, axis=2)
                    yield video_clip

        # video clip paths
        if self._phase == 'train':
            dataset = tf.data.Dataset.from_generator(generator=train_video_clip_generator,
                                                     output_types=tf.float32,
                                                     output_shapes=[resize_height, resize_width, clip_length * 3])
        else:
            dataset = tf.data.Dataset.from_generator(generator=test_video_clip_generator,
                                                     output_types=tf.float32,
                                                     output_shapes=[resize_height, resize_width, clip_length * 3])

        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=500)
        dataset = dataset.shuffle(buffer_size=500).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, video_name):
        assert video_name in self.videos.keys(), 'video = {} is not in {}!'.format(video_name, self.videos.keys())
        return self.videos[video_name]

    def setup(self):
        # videos = glob.glob(os.path.join(self.dir, '*'))
        videos = []
        normal_classes, abnorm_classes = get_exp_classes(self._split, ntu=self._ntu)
        # c_names = [dir for dir in next(os.walk(self.dir))[1]]

        c_names = get_kinetics_names(ntu=self._ntu)

        for c_idx, c_dir in enumerate(c_names):
            # if not os.path.isdir(os.path.join(self.dir, c_dir)):
            #     continue

            if self._phase == 'test':
                if (not c_idx in normal_classes) and (not c_idx in abnorm_classes):
                    continue
            elif self._phase == 'train':
                if self._inv:
                    if not c_idx in abnorm_classes:
                        continue
                else:
                    # if self._phase == 'train':
                    if not c_idx in normal_classes:
                        continue

            for vid in os.listdir(os.path.join(self.dir, c_dir)):
                videos.append(os.path.join(c_dir, vid))
        for video in sorted(videos):
            video_name = video.split('/')[-1]

            self.videos[video_name] = {}
            self.videos[video_name]['path'] = os.path.join(self.dir,video)
            # self.videos[video_name]['length'] =
            # self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            # self.videos[video_name]['frame'].sort()
            cap = cv2.VideoCapture(self.videos[video_name]['path'])
            vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.videos[video_name]['length'] = vid_length
        print("number of videos {}".format(len(self.videos)))


    def get_video_clips(self, video, start, end):
        # assert video in self.videos, 'video = {} must in {}!'.format(video, self.videos.keys())
        # assert start >= 0, 'start = {} must >=0!'.format(start)
        # assert end <= self.videos[video]['length'], 'end = {} must <= {}'.format(video, self.videos[video]['length'])
        video_info = self.videos[video]
        cap = cv2.VideoCapture(video_info['path'])
        vid_length = video_info['length']
        # print("vid: {} - {} - {}".format(video_info['path'], vid_length, resize_width))
        if not vid_length:
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        # print("starting from frame: {}".format(start))
        batch = []
        for frame_id in range(start, end):
            ret, frame = cap.read()
            if ret:
                frame_resize = np_load_frame(frame, self._resize_height, self._resize_width)
                # print("vid: {} read complete".format(video_info['path']))
                batch.append(frame_resize)
            else:
                print('frame not loaded')
        return np.concatenate(batch, axis=2)
        # yield video_clip


        #
        # batch = []
        # for i in range(start, end):
        #     image = np_load_frame(self.videos[video]['frame'][i], self._resize_height, self._resize_width)
        #     batch.append(image)
        #
        # return np.concatenate(batch, axis=2)


def log10(t):
    """
    Calculates the base-10 log of each element in t.

    @param t: The tensor from which to calculate the base-10 log.

    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr_error(gen_frames, gt_frames):
    """
    Computes the Peak Signal to Noise Ratio error between the generated images and the ground
    truth images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.

    @return: A scalar tensor. The mean Peak Signal to Noise Ratio error over each frame in the
             batch.
    """
    shape = tf.shape(gen_frames)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])
    gt_frames = (gt_frames + 1.0) / 2.0
    gen_frames = (gen_frames + 1.0) / 2.0
    square_diff = tf.square(gt_frames - gen_frames)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(square_diff, [1, 2, 3])))
    return tf.reduce_mean(batch_errors)


def diff_mask(gen_frames, gt_frames, min_value=-1, max_value=1):
    # normalize to [0, 1]
    delta = max_value - min_value
    gen_frames = (gen_frames - min_value) / delta
    gt_frames = (gt_frames - min_value) / delta

    gen_gray_frames = tf.image.rgb_to_grayscale(gen_frames)
    gt_gray_frames = tf.image.rgb_to_grayscale(gt_frames)

    diff = tf.abs(gen_gray_frames - gt_gray_frames)
    return diff


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')




