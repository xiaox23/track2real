import copy

import cv2
import numpy as np
import math


def transform_label_to_real(label):
    transformed_label = np.zeros_like(label)
    transformed_label[..., 0] = label[..., 1]
    transformed_label[..., 1] = -label[..., 0]
    transformed_label[..., 2] = label[..., 2]
    return transformed_label


def transform_label_to_sim(label):
    transformed_label = np.zeros_like(label)
    transformed_label[..., 0] = -label[..., 1]
    transformed_label[..., 1] = label[..., 0]
    transformed_label[..., 2] = label[..., 2]
    return transformed_label


def vertical_ray_intersects_segment(ray_point, segment_start, segment_end):
    # vertical and upward
    # include start point but exclude end point
    if segment_start[0] == segment_end[0]:
        return False
    segment_slope = (segment_end[1] - segment_start[1]) / (segment_end[0] - segment_start[0])
    segment_b = segment_end[1] - segment_end[0] * segment_slope

    intersect_point = (ray_point[0], ray_point[0] * segment_slope + segment_b)
    if intersect_point[1] >= ray_point[1]:
        intersect_ratio = (ray_point[0] - segment_start[0]) / (segment_end[0] - segment_start[0])
        if 0 <= intersect_ratio < 1:
            return True
        else:
            return False
    else:
        return False


def point_in_polygon(point, polygon_points):
    polygon_point_num = len(polygon_points)
    intersect_num = 0
    for i in range(polygon_point_num):
        seg_start = polygon_points[i]
        seg_end = polygon_points[(i + 1) % polygon_point_num]
        if vertical_ray_intersects_segment(point, seg_start, seg_end):
            intersect_num += 1

    if intersect_num % 2 == 0:
        return False
    else:
        return True


def line_intersection(A, B, C, D):
    xdiff = (A[0] - B[0], C[0] - D[0])
    ydiff = (A[1] - B[1], C[1] - D[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    def ccw(_A, _B, _C):
        return (_C[1] - _A[1]) * (_B[0] - _A[0]) > (_B[1] - _A[1]) * (_C[0] - _A[0])

    # Return true if line segments AB and CD intersect
    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    if intersect(A, B, C, D):
        d = (det(A, B), det(C, D))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
    else:
        return None


def get_intersection_point_with_polygon(segment_start, segment_end, polygon_points):
    num_polygon_point = len(polygon_points)
    for i in range(num_polygon_point):
        point_1 = polygon_points[i]
        point_2 = polygon_points[(i + 1) % num_polygon_point]

        intersect_result = line_intersection(segment_start, segment_end, point_1, point_2)
        if intersect_result is None:
            continue
        else:
            return intersect_result
    return None


def compute_polygon_area(points):
    point_num = len(points)
    if point_num < 3: return 0.0
    s = points[0][1] * (points[point_num - 1][0] - points[1][0])
    # for i in range(point_num): # (int i = 1 i < point_num ++i):
    for i in range(1, point_num):
        s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
    return abs(s / 2.0)


def get_contact_state(peg_points, hole_points):
    """
    :param peg_points: list of tuple or list of list, should be ordered
    :param hole_points: list of tuple or list of list, should be ordered
    :return: list of tuple or list of list, each element is a polygonal contact region
    """

    # first find a start point that is in the hole
    num_peg_points = len(peg_points)
    start_peg_point = None
    for i in range(num_peg_points):
        if point_in_polygon(peg_points[i], hole_points):
            start_peg_point = i
            break

    if start_peg_point is None:
        return None, None, False

    # walk through all peg points, direction 1
    rotation_region_direction_1 = []
    intersection_point_1, intersection_point_2 = None, None
    for walk_through_index in range(start_peg_point, start_peg_point + num_peg_points):
        valid_index = walk_through_index % num_peg_points
        valid_index_next = (valid_index + 1) % num_peg_points
        start_point = peg_points[valid_index]
        rotation_region_direction_1.append(copy.deepcopy(start_point))
        end_point = peg_points[valid_index_next]
        if not point_in_polygon(end_point, hole_points):
            intersection_point_1 = get_intersection_point_with_polygon(start_point, end_point, hole_points)
            rotation_region_direction_1.append(copy.deepcopy(intersection_point_1))
            break

    # walk through all peg points, direction 2
    rotation_region_direction_2 = []
    for walk_through_index in range(start_peg_point, start_peg_point - num_peg_points, -1):
        valid_index = walk_through_index % num_peg_points
        valid_index_next = (valid_index - 1) % num_peg_points
        start_point = peg_points[valid_index]
        rotation_region_direction_2.append(copy.deepcopy(start_point))
        end_point = peg_points[valid_index_next]
        if not point_in_polygon(end_point, hole_points):
            intersection_point_2 = get_intersection_point_with_polygon(start_point, end_point, hole_points)
            rotation_region_direction_2.append(copy.deepcopy(intersection_point_2))
            break

    rotation_region = rotation_region_direction_1 + rotation_region_direction_2[::-1][:-1]
    hole_area = compute_polygon_area(peg_points)
    rotation_area = compute_polygon_area(rotation_region)
    center_outside_contact = (rotation_area > hole_area / 2)

    return intersection_point_1, intersection_point_2, center_outside_contact


def generate_rectangle(center, size, theta, rotation_first=False):
    center_x, center_y = center
    x, y = size
    v = np.array([
        [- x / 2, - y / 2],
        [x / 2, - y / 2],
        [x / 2, y / 2],
        [- x / 2, y / 2]
    ])
    rot = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])  # the u-v-z axis is not right-handed
    if not rotation_first:
        v_rotated = (rot @ v.T).T + np.array([center_x, center_y])
    else:
        v_rotated = (rot @ (v + np.array([center_x, center_y])).T).T
    return v_rotated.tolist()


def get_point_to_line_distance(point: tuple, line_endpoint_1: tuple, line_endpoint_2: tuple):
    x1, y1 = line_endpoint_1
    x2, y2 = line_endpoint_2
    a, b = point
    A = x1 ** 2 + x2 ** 2 + y1 ** 2 + y2 ** 2 - 2 * x1 * x2 - 2 * y1 * y2
    B = -2 * (x2 ** 2 + y2 ** 2 - x1 * x2 - y1 * y2 + a * (x1 - x2) + b * (y1 - y2))
    C = a ** 2 + b ** 2 - 2 * a * x2 - 2 * b * y2 + x2 ** 2 + y2 ** 2
    min_alpha = - B / 2 / A
    closet_point = x1 * min_alpha + x2 * (1 - min_alpha), y1 * min_alpha + y2 * (1 - min_alpha)
    min_distance = np.sqrt(A * min_alpha ** 2 + B * min_alpha + C)
    # print(point, line_endpoint_1, line_endpoint_2, min_distance)
    return min_distance, closet_point


def check_blocked(offset, clearance=2.5, margin=0):
    peg_points = generate_rectangle((offset[0], offset[1]), (40, 30), offset[2])
    hole_points = generate_rectangle((0, 0), (40 + 2 * clearance + margin, 30 + 2 * clearance + margin), 0)
    num_peg_points = len(peg_points)

    point_inside_num = 0
    for i in range(num_peg_points):
        if point_in_polygon(peg_points[i], hole_points):
            point_inside_num += 1

    if point_inside_num == num_peg_points:
        return False
    else:
        return True


def convert_offset_to_contact_line(offset):
    """
    :param offset: x_offset (mm), y_offset (mm), theta_offset (radian)
    :return: contact line params, (distance * cosine(angle), distance * sine(angle), inside or outside contact)
    """
    peg_points = generate_rectangle((offset[0], offset[1]), (40, 30), offset[2])
    hole_points = generate_rectangle((0, 0), (45, 35), 0)
    contact_point_1, contact_point_2, center_outside_contact = get_contact_state(peg_points, hole_points)

    if contact_point_1 is None or contact_point_2 is None:
        return 0, 0, 0

    peg_center = (offset[0], offset[1])
    peg_unit_vector = (math.cos(offset[2]), math.sin(offset[2]))

    dist, closest_point = get_point_to_line_distance(peg_center, contact_point_1, contact_point_2)
    orthogonal_vector = closest_point[0] - peg_center[0], closest_point[1] - peg_center[1]
    angle = math.atan2(orthogonal_vector[1], orthogonal_vector[0]) - math.atan2(peg_unit_vector[1],
                                                                                peg_unit_vector[0])

    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi

    result = dist * math.cos(angle), dist * math.sin(angle), 2 * int(center_outside_contact) - 1
    return result


def adapt_marker_seq_to_unified_size(input_marker_seq, desired_size):
    # (time_step, num_of_points, x_y)
    original_point_num = input_marker_seq.shape[1]
    if original_point_num >= desired_size:
        ret = input_marker_seq[:, :desired_size, ...]
    else:
        ret = np.zeros((input_marker_seq.shape[0], desired_size, 2))
        ret[:, :original_point_num, :] = input_marker_seq.copy()
        ret[:, original_point_num:, :] = ret[:, original_point_num - 1:original_point_num, :]
    return ret

normalize_factor_contact_line = np.array([25, 25, 1])


def generate_offset_grid(max_offset, num_per_axis: int=6, clearance=2.5, margin=0):
    max_x = max_offset[0]
    max_y = max_offset[1]
    max_theta = max_offset[2]

    offsets = []
    for x_i in range(num_per_axis):
        x_offset = -max_x + max_x * 2 / (num_per_axis - 1) * x_i
        for y_i in range(num_per_axis):
            y_offset = -max_x + max_y * 2 / (num_per_axis - 1) * y_i
            for theta_i in range(num_per_axis):
                theta_offset = -max_theta + max_theta * 2 / (num_per_axis - 1) * theta_i
                if check_blocked((x_offset, y_offset, theta_offset * np.pi / 180), clearance=clearance, margin=margin):
                    offsets.append([x_offset, y_offset, theta_offset])
                # else:
                #     print([x_offset, y_offset, theta_offset])

    return offsets


def generate_offset_grid_with_num(max_offset, num, margin=0):
    offset_grid = generate_offset_grid(max_offset, 10, margin=margin)
    interval = len(offset_grid) / num
    test_offset_list = []
    for i in range(num):
        cur_id = math.floor(i * interval)
        test_offset_list.append(offset_grid[cur_id])
    return test_offset_list


def get_tactile_flow_map(marker_flow_points):
    marker_pos = marker_flow_points[0, :]
    marker_displacement = marker_flow_points[1, :] - marker_flow_points[0, :]
    marker_flow_map = np.zeros((8, 8, 2))
    interval = 320 / 9.6
    for row_id in range(8):
        for col_id in range(8):
            marker_v_range = np.array([row_id * interval, (row_id + 1) * interval]) + interval
            marker_u_range = 320 - (np.array([(col_id + 1) * interval, col_id * interval]) + 0.6 * interval)
            marker_u_valid = np.logical_and(marker_u_range[0] < marker_pos[:, 0],  marker_pos[:, 0]< marker_u_range[-1])
            marker_v_valid = np.logical_and(marker_v_range[0] < marker_pos[:, 1],  marker_pos[:, 1]< marker_v_range[-1])
            marker_idx_mask = np.logical_and(marker_u_valid, marker_v_valid)
            marker_idx = np.where(marker_idx_mask)[0]  # np.where returns a tuple. first element is np.array
            # print(marker_idx)
            if marker_idx.size > 0:
                marker_idx = marker_idx[0]
                marker_flow_map[row_id, col_id, 0] = marker_displacement[marker_idx, 1]
                marker_flow_map[row_id, col_id, 1] = -marker_displacement[marker_idx, 0]


    return marker_flow_map


def get_double_side_tactile_map(marker_flow_obs):
    left_flow = get_tactile_flow_map(marker_flow_obs[1])
    right_flow = get_tactile_flow_map(marker_flow_obs[0])
    return left_flow, right_flow


def normalize_tactile_flow_map(left_flow, right_flow):
    left_lengths = np.linalg.norm(left_flow, axis=-1)
    right_lengths = np.linalg.norm(right_flow, axis=-1)

    max_length = max(np.max(left_lengths), np.max(right_lengths)) + 1e-5
    normalized_left_flow = left_flow / (max_length / 30.0)
    normalized_right_flow = right_flow / (max_length / 30.0)

    return normalized_left_flow, normalized_right_flow


def visualize_tactile(tactile_array):
    resolution = 20
    horizontal_space = 20
    vertical_space = 40
    nrows = tactile_array.shape[1]
    ncols = tactile_array.shape[2]

    imgs_tactile = np.zeros(
        (ncols * resolution * 2 + vertical_space * 3, nrows * resolution + horizontal_space * 2, 3),
        dtype=float)

    for finger_idx in range(2):
        for row in range(nrows):
            for col in range(ncols):
                loc0_x = row * resolution + resolution // 2 + horizontal_space
                loc0_y = col * resolution + resolution // 2 + finger_idx * ncols * resolution + finger_idx * vertical_space + vertical_space
                loc1_x = loc0_x + tactile_array[finger_idx * 2][row, col]
                loc1_y = loc0_y + tactile_array[finger_idx * 2 + 1][row, col]
                cv2.arrowedLine(imgs_tactile, (int(loc0_x), int(loc0_y)), (int(loc1_x), int(loc1_y)),
                                    (0.0, 1.0, 0.0), 2, tipLength=0.3)

    return imgs_tactile