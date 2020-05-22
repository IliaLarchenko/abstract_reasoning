import gc
import time

import numpy as np

from src.preprocessing import get_color, get_color_scheme, get_dict_hash, get_mask_from_block_params


def paint_mask(sample, rotate_target=0):
    target_image = np.rot90(np.uint8(sample["train"][0]["output"]), rotate_target)
    unique = np.unique(target_image)
    if len(unique) > 2:
        return 3, None
    t_n, t_m = target_image.shape
    candidates = []
    max_time = 300
    start_time = time.time()
    for mask in sample["train"][0]["masks"]:
        if time.time() - start_time > max_time:
            break
        if t_n == mask["mask"].shape[0] and t_m == mask["mask"].shape[1]:
            unique = np.unique(target_image[mask["mask"]])
            if len(unique) != 1:
                continue
            color2 = unique[0]
            unique = np.unique(target_image[np.logical_not(mask["mask"])])
            if len(unique) != 1:
                continue
            color1 = unique[0]
            for color_dict1 in sample["train"][0]["colors"][color1].copy():
                for color_dict2 in sample["train"][0]["colors"][color2].copy():
                    candidates.append(
                        {
                            "mask": {"params": mask["params"], "operation": mask["operation"]},
                            "color1": color_dict1.copy(),
                            "color2": color_dict2.copy(),
                        }
                    )
    gc.collect()

    for k in range(1, len(sample["train"])):
        start_time = time.time()
        original_image = np.uint8(sample["train"][k]["input"])
        target_image = np.rot90(np.uint8(sample["train"][k]["output"]), rotate_target)
        t_n, t_m = target_image.shape
        new_candidates = []
        if "block_cache" not in sample["train"][k]:
            sample["train"][k]["block_cache"] = {}
        if "mask_cache" not in sample["train"][k]:
            sample["train"][k]["mask_cache"] = {}

        for candidate in candidates:
            if time.time() - start_time > max_time:
                break
            status, mask = get_mask_from_block_params(
                original_image,
                candidate["mask"],
                block_cache=sample["train"][k]["block_cache"],
                color_scheme=sample["train"][k],
                mask_cache=sample["train"][k]["mask_cache"],
            )
            if status != 0 or mask.shape[0] != t_n or mask.shape[1] != t_m:
                continue
            color1 = get_color(candidate["color1"], sample["train"][k]["colors"])
            color2 = get_color(candidate["color2"], sample["train"][k]["colors"])

            if color1 < 0 or color2 < 0:
                continue

            part1 = (1 - mask) * color1
            part2 = mask * color2
            result = part1 + part2
            result == target_image
            if (target_image == ((1 - mask) * color1 + mask * color2)).all():
                new_candidates.append(candidate)
        candidates = new_candidates.copy()
        del sample["train"][k]["mask_cache"]
        del sample["train"][k]["block_cache"]
        gc.collect()

    if len(candidates) == 0:
        return 1, None

    answers = []
    for _ in sample["test"]:
        answers.append([])

    result_generated = False
    for test_n, test_data in enumerate(sample["test"]):
        original_image = np.uint8(test_data["input"])
        if "block_cache" not in sample["test"][test_n]:
            sample["train"][test_n]["block_cache"] = {}
        if "mask_cache" not in sample["test"][test_n]:
            sample["train"][test_n]["mask_cache"] = {}
        color_scheme = get_color_scheme(original_image)
        for candidate in candidates:
            status, mask = get_mask_from_block_params(
                original_image,
                candidate["mask"],
                color_scheme=color_scheme,
                block_cache=sample["train"][test_n]["block_cache"],
                mask_cache=sample["train"][test_n]["mask_cache"],
            )
            if status != 0:
                continue
            color1 = get_color(candidate["color1"], color_scheme["colors"])
            color2 = get_color(candidate["color2"], color_scheme["colors"])

            if color1 < 0 or color2 < 0:
                continue
            prediction = ((1 - mask) * color1) + (mask * color2)
            answers[test_n].append(np.rot90(prediction, k=-rotate_target))
            result_generated = True

    if result_generated:
        return 0, answers
    else:
        return 2, None


def filter_list_of_dicts(list1, list2):
    set_of_hashes = {get_dict_hash(item1) for item1 in list1}
    final_list = []
    for item2 in list2:
        if get_dict_hash(item2) in set_of_hashes:
            final_list.append(item2)
    return final_list


def several_colors_square(sample):
    color_candidates_final = []

    for k in range(len(sample["train"])):
        color_candidates = []
        target_image = np.uint8(sample["train"][k]["output"])
        if target_image.shape[0] != target_image.shape[1]:
            return 1, None
        size = target_image.shape[0]
        if size > sample["train"][k]["colors_num"]:
            return 2, None

        size_diff = sample["train"][k]["colors_num"] - size
        for i in range(size_diff + 1):
            colors_array = np.zeros((size, size))
            for j in range(size):
                colors_array[j:-j] = sample["train"][k]["colors_sorted"][i + j]
            if (colors_array == target_image).all():
                color_candidates.append({"type": "square", "i": i, "direct": 0, "size_diff": size_diff})

            for j in range(size):
                colors_array[j:-j] = sample["train"][k]["colors_sorted"][::-1][i + j]
            if (colors_array == target_image).all():
                color_candidates.append({"type": "square", "i": i, "direct": 1, "size_diff": size_diff})

        if k == 0:
            color_candidates_final = color_candidates
        else:
            color_candidates_final = filter_list_of_dicts(color_candidates, color_candidates_final)
        if len(color_candidates_final) == 0:
            return 2, None

    answers = []
    for _ in sample["test"]:
        answers.append([])

    result_generated = False
    for test_n, test_data in enumerate(sample["test"]):
        original_image = np.uint8(test_data["input"])
        color_scheme = get_color_scheme(original_image)
        for result_dict in color_candidates_final:
            i = result_dict["i"]
            rotate = result_dict["rotate"]
            size = color_scheme["colors_num"] - size_diff
            prediction = np.zeros((size, size))
            for j in range(size):
                if result_dict["direct"] == 0:
                    prediction[j:-j] = sample["train"][k]["colors_sorted"][i + j]
                else:
                    prediction[j:-j] = sample["train"][k]["colors_sorted"][::-1][i + j]

            answers[test_n].append(prediction)
            result_generated = True

    if result_generated:
        return 0, answers
    else:
        return 3, None


def swap_two_colors(image):
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
