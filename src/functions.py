import numpy as np
from src.preprocessing import (
    get_color,
    get_color_scheme,
    get_mask_from_block_params,
    get_dict_hash,
)
import time
import gc


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
                            "mask": {
                                "params": mask["params"],
                                "operation": mask["operation"],
                            },
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
    final_list = []
    for item2 in list2:
        for item1 in list1:
            if get_dict_hash(item1) == get_dict_hash(item2):
                final_list.append(item1)
                break
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
                color_candidates.append(
                    {"type": "square", "i": i, "direct": 0, "size_diff": size_diff}
                )

            for j in range(size):
                colors_array[j:-j] = sample["train"][k]["colors_sorted"][::-1][i + j]
            if (colors_array == target_image).all():
                color_candidates.append(
                    {"type": "square", "i": i, "direct": 1, "size_diff": size_diff}
                )

        if k == 0:
            color_candidates_final = color_candidates
        else:
            color_candidates_final = filter_list_of_dicts(
                color_candidates, color_candidates_final
            )
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
