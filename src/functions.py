import numpy as np

from src.preprocessing import get_color_scheme, get_dict_hash


def filter_list_of_dicts(list1, list2):
    """Returns the intersection of two lists of dicts"""
    set_of_hashes = {get_dict_hash(item1) for item1 in list1}
    final_list = []
    for item2 in list2:
        if get_dict_hash(item2) in set_of_hashes:
            final_list.append(item2)
    return final_list


def swap_two_colors(image):
    """sawaps two colors"""
    unique = np.unique(image)
    if len(unique) != 2:
        return 1, None
    result = image.copy()
    result[image == unique[0]] = unique[1]
    result[image == unique[1]] = unique[0]
    return 0, result


def combine_two_lists(list1, list2):
    result = list1.copy()
    for item2 in list2:
        exist = False
        for item1 in list1:
            if (item2 == item1).all():
                exist = True
                break
        if not exist:
            result.append(item2)
    return result


def intersect_two_lists(list1, list2):
    """ intersects two lists of np.arrays"""
    result = []
    for item2 in list2:
        for item1 in list1:
            if (item2.shape == item1.shape) and (item2 == item1).all():
                result.append(item2)
                break
    return result


def check_surface_block(image, i, j, block):
    color = 11
    b = (image.shape[0] - i) // block.shape[0] + int(((image.shape[0] - i) % block.shape[0]) > 0)
    r = (image.shape[1] - j) // block.shape[1] + int(((image.shape[1] - j) % block.shape[1]) > 0)
    t = (i) // block.shape[0] + int((i) % block.shape[0] > 0)
    l = (j) // block.shape[1] + int((j) % block.shape[1] > 0)

    full_image = np.ones(((b + t) * block.shape[0], (r + l) * block.shape[1])) * color
    start_i = (block.shape[0] - i) % block.shape[0]
    start_j = (block.shape[1] - j) % block.shape[1]

    full_image[start_i : start_i + image.shape[0], start_j : start_j + image.shape[1]] = image

    blocks = []
    for k in range(b + t):
        for n in range(r + l):
            new_block = full_image[
                k * block.shape[0] : (k + 1) * block.shape[0], n * block.shape[1] : (n + 1) * block.shape[1]
            ]
            mask = np.logical_and(new_block != color, block != color)
            if (new_block == block)[mask].all():
                blocks.append(new_block)
            else:
                return 1, None

    new_block = block.copy()
    for curr_block in blocks:
        mask = np.logical_and(new_block != color, curr_block != color)
        if (new_block == curr_block)[mask].all():
            new_block[new_block == color] = curr_block[new_block == color]
        else:
            return 2, None

    if (new_block == color).any():
        return 3, None

    return 0, new_block


def find_mosaic_block(image, params):
    """ predicts 1 output image given input image and prediction params"""
    itteration_list1 = list(range(2, sum(image.shape) - 3))
    if params["big_first"]:
        itteration_list1 = itteration_list1[::-1]
    for size in itteration_list1:
        if params["direction"] == "all":
            itteration_list = list(range(1, size))
        elif params["direction"] == "vert":
            itteration_list = [image.shape[0]]
        else:
            itteration_list = [size - image.shape[1]]
        for i_size in itteration_list:
            j_size = size - i_size
            if j_size < 1 or i_size < 1:
                continue
            block = image[0 : 0 + i_size, 0 : 0 + j_size]
            status, predict = check_surface_block(image, 0, 0, block)
            if status != 0:
                continue
            return 0, predict

    return 1, None


def reconstruct_mosaic_from_block(block, params, original_image=None):
    if params["mosaic_size_type"] == "fixed":
        temp_shape = [0, 0]
        temp_shape[0] = params["mosaic_shape"][0] + params["mosaic_shape"][0] % block.shape[0]
        temp_shape[1] = params["mosaic_shape"][1] + params["mosaic_shape"][1] % block.shape[1]

        result = np.zeros(temp_shape)
        for i in range(temp_shape[0] // block.shape[0]):
            for j in range(temp_shape[1] // block.shape[1]):
                result[
                    i * block.shape[0] : (i + 1) * block.shape[0], j * block.shape[1] : (j + 1) * block.shape[1]
                ] = block
        result = result[: params["mosaic_shape"][0], : params["mosaic_shape"][1]]
    elif params["mosaic_size_type"] == "size":
        result = np.zeros((params["mosaic_size"][0] * block.shape[0], params["mosaic_size"][1] * block.shape[1]))
        for i in range(params["mosaic_size"][0]):
            for j in range(params["mosaic_size"][1]):
                result[
                    i * block.shape[0] : (i + 1) * block.shape[0], j * block.shape[1] : (j + 1) * block.shape[1]
                ] = block
    elif params["mosaic_size_type"] == "same":
        params = params.copy()
        params["mosaic_shape"] = original_image.shape
        params["mosaic_size_type"] = "fixed"
        result = reconstruct_mosaic_from_block(block, params, original_image=None)
    elif params["mosaic_size_type"] == "same_rotated":
        params = params.copy()
        params["mosaic_shape"] = original_image.T.shape
        params["mosaic_size_type"] = "fixed"
        result = reconstruct_mosaic_from_block(block, params, original_image=None)
    elif params["mosaic_size_type"] == "color_num":
        params = params.copy()
        color_num = len(np.unique(original_image))
        params["mosaic_size"] = [color_num, color_num]
        params["mosaic_size_type"] = "size"
        result = reconstruct_mosaic_from_block(block, params, original_image=None)
    elif params["mosaic_size_type"] == "block_shape_size":
        params = params.copy()
        color_num = len(np.unique(original_image))
        params["mosaic_size"] = block.shape
        params["mosaic_size_type"] = "size"
        result = reconstruct_mosaic_from_block(block, params, original_image=None)
    else:
        return None
    return result
