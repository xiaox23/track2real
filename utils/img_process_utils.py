import numpy as np
import cv2
import copy

import torch
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import transforms


def get_duplicate_id(arr):
    """
    return: list of tuples like (1,[0,1,2]), meaning that index 0,1,2 all have value of 1
    """
    uniq = np.unique(arr).tolist()
    ret = []

    def check_id(target, a):
        b = []
        for index, nums in enumerate(a):
            if nums == target:
                b.append(index)
        return (b)

    for i in range(len(uniq)):
        ret.append([])
    for index, nums in enumerate(arr):
        id = uniq.index(nums)
        ret[id].append(index)

    ans = [(uniq[i], ret[i]) for i in range(len(uniq))]
    return ans


def get_mapping(prev_markers, markers, max_distance=30):
    """获取两帧的关键点之间的映射及丢失情况"""
    prev_markers = copy.deepcopy(prev_markers)
    markers = copy.deepcopy(markers)

    prev_markers_array = np.array(prev_markers)
    markers = np.array(markers)
    if prev_markers_array.ndim <= 1:
        return np.zeros((markers.shape[0],)).astype(np.int), np.ones((markers.shape[0],)).astype(np.bool),
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(prev_markers_array)
    distances, indices = nbrs.kneighbors(markers)
    lost = distances > max_distance
    distances = distances.flatten()
    mapping = indices.flatten()
    lost = lost.flatten()

    dup = get_duplicate_id(mapping)
    for (value, index_list) in dup:
        min_id = np.argmin(distances[index_list])
        for duplicated in index_list:
            if duplicated != index_list[min_id]:
                lost[duplicated] = True

    return mapping, lost


def _mask_marker(image):
    blur_size_up = 13
    blur_size_down = 7
    blur_size_diff = 5
    marker_threshold = 1
    row, col = image.shape[:2]
    image = image.astype(np.float32)
    blur = cv2.GaussianBlur(image, (blur_size_up, blur_size_up), 0)
    blur2 = cv2.GaussianBlur(image, (blur_size_down, blur_size_down), 0)
    diff = blur - blur2
    diff *= 16.0
    diff = np.clip(diff, 0., 255.0)
    diff = cv2.GaussianBlur(diff, (blur_size_diff, blur_size_diff), 0)
    if len(diff.shape) == 3:
        marker_mask = diff.sum(-1) > 255 * marker_threshold
    else:
        marker_mask = diff > 255 * marker_threshold / 3
    marker_mask = marker_mask.astype(np.uint8) * 255
    # cv2.imshow("markers", marker_mask)
    # cv2.waitKey(1)
    return marker_mask  # 255 for markers


class TactileMarkerTracker:
    def __init__(self, frame_image):
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 1
        params.maxThreshold = 12
        params.minDistBetweenBlobs = 12
        params.filterByArea = True
        params.minArea = 9
        params.maxArea = 200
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        self.marker_detector = cv2.SimpleBlobDetector_create(params)

        marker_mask = _mask_marker(frame_image)
        self.init_markers = self._get_markers(marker_mask)
        self.total_mapping = np.arange(len(self.init_markers))
        self.total_lost = np.zeros((len(self.init_markers),)).astype(np.bool)
        self.last_markers = self.init_markers
        self.tracking_max_distance = 2.5 / 0.04 / 2

    def _get_markers(self, marker_mask_image):
        marker_points = self.marker_detector.detect(255 - marker_mask_image)
        marker_points = [(a.pt[0], a.pt[1]) for a in marker_points]
        return marker_points

    def track_marker(self, cropped_image):
        curr_markers = self._get_markers(_mask_marker(cropped_image))
        curr_mapping, curr_lost = get_mapping(self.last_markers, curr_markers, self.tracking_max_distance)

        self.total_mapping = self.total_mapping[curr_mapping]
        self.total_lost = self.total_lost[curr_mapping] | curr_lost
        self.last_markers = curr_markers

        return curr_markers


def get_valid_marker_sequence(img_seq: np.array):
    # img_seq shape: 15, 320, 320, 3
    # img_seq = img_seq[:, 80:400, 160:480, :]
    marker_flow_seq = []
    marker_tracker = TactileMarkerTracker(img_seq[0])
    marker_flow_seq.append((marker_tracker.last_markers, marker_tracker.total_mapping, marker_tracker.total_lost))
    for i in range(1, img_seq.shape[0]):
        marker_tracker.track_marker(img_seq[i])
        marker_flow_seq.append((marker_tracker.last_markers, marker_tracker.total_mapping, marker_tracker.total_lost))

    ret_seq = []
    not_lost_marker_ids = sorted([marker_tracker.total_mapping[i] if not marker_tracker.total_lost[i] else -1 for i in
                                  range(marker_tracker.total_lost.shape[0])])

    for time_step in range(0, img_seq.shape[0]):
        not_lost_marker_in_cur_step = []
        for not_lost_marker_id in not_lost_marker_ids:  # 按顺序遍历寻找没有丢失的marker， 以保证顺序相同
            for marker_id in range(len(marker_flow_seq[time_step][0])):
                if marker_flow_seq[time_step][1][marker_id] == not_lost_marker_id and not marker_flow_seq[time_step][2][
                    marker_id]:
                    not_lost_marker_in_cur_step.append(marker_flow_seq[time_step][0][marker_id])

        ret_seq.append(np.array(not_lost_marker_in_cur_step))

    return np.stack(ret_seq, axis=0)


def get_marker_mapping(_from, _to):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(_from)
    distances, indices = nbrs.kneighbors(_to)
    return indices


def get_marker_spacing(marker_array, direction="bottom"):
    # choose markers on the bottom line
    marker_inside = marker_array[marker_array[:, 1] <= 315]
    marker_inside = marker_inside[marker_inside[:, 1] >= 5]
    marker_inside = marker_inside[marker_inside[:, 0] <= 270]
    marker_inside = marker_inside[marker_inside[:, 0] >= 50]

    if direction == "bottom":
        max_v = np.max(marker_inside[:, 1])
        marker_chosen_line = marker_inside[marker_inside[:, 1] > max_v - 16]
    elif direction == "top":
        min_v = np.min(marker_inside[:, 1])
        marker_chosen_line = marker_inside[marker_inside[:, 1] < min_v + 16]
    marker_chosen_line_sorted = marker_chosen_line[
        np.argsort(marker_chosen_line[:, 0] * 320 + marker_chosen_line[:, 1])]
    diff = np.diff(marker_chosen_line_sorted, axis=0)
    distance = np.sqrt(np.sum(diff ** 2, axis=1))
    # print(distance.shape)
    line_length = np.sqrt(np.sum((marker_chosen_line_sorted[-1] - marker_chosen_line_sorted[0]) ** 2))
    return np.mean(distance), line_length


def get_marker_spacing_grid(marker_array, max_distance=40):
    # choose markers on the bottom line
    marker_num = marker_array.shape[0]
    marker_list = [marker_array[i] for i in range(marker_num)]
    distance_sum = 0
    distance_num = 0
    for i in range(marker_num):
        for j in range(i + 1, marker_num):
            dist = np.sqrt(np.sum((marker_list[j] - marker_list[i]) ** 2))
            if dist < max_distance:
                distance_sum += dist
                distance_num += 1
    avg_distance = distance_sum / distance_num
    print(f"Number of adjacent pairs: {distance_num}, average distance: {avg_distance}.")
    return avg_distance, distance_num


def convert_to_binary_torch(_input: np.array):
    transform_to_tensor = transforms.ToTensor()
    output = []
    original_shape = _input.shape
    if len(original_shape) == 5:
        _input = _input.reshape(
            (original_shape[0] * original_shape[1], original_shape[2], original_shape[3], original_shape[4]))
    num_of_imgs = _input.shape[0]
    with torch.no_grad():
        for i in range(num_of_imgs):
            cur_img = _input[i, ...]
            cur_img = cv2.resize(cur_img, (256, 256))
            if len(cur_img.shape) == 2:
                cur_img = np.expand_dims(cur_img, axis=-1)
            cur_img = _mask_marker(cur_img)
            cur_img = transform_to_tensor(cur_img)
            cur_img = cur_img.type(torch.bool)
            output.append(cur_img)
        output = torch.stack(output)
        if len(original_shape) == 5:
            output = output.reshape(original_shape[0], original_shape[1], 1, 256, 256)
    return output


def create_position_encoding(img_size=256):
    with torch.no_grad():
        angle = torch.linspace(0, 2 * torch.pi, steps=img_size, dtype=torch.float32)
        u_cos = torch.reshape(torch.cos(angle), (1, img_size)).repeat((img_size, 1))
        u_sin = torch.reshape(torch.sin(angle), (1, img_size)).repeat((img_size, 1))

        v_cos = torch.reshape(torch.cos(angle), (img_size, 1)).repeat((1, img_size))
        v_sin = torch.reshape(torch.sin(angle), (img_size, 1)).repeat((1, img_size))
        position_encoding = torch.stack([u_cos, u_sin, v_cos, v_sin])
    return position_encoding

