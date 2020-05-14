import numpy as np
from src.preprocessing import (
    get_predict,
    get_color,
    get_color_scheme,
    get_mask_from_block_params,
    get_dict_hash,
)
import time
import gc


def mask_to_blocks(sample, rotate_target=0, num_masks=1):
    target_image = np.rot90(np.uint8(sample["train"][0]["output"]), rotate_target)
    t_n, t_m = target_image.shape
    candidates = []
    max_time = 300
    start_time = time.time()
    for block in sample["train"][0]["blocks"]:
        if len(block["params"]) > 0 and block["params"][-1]["type"] == "color_swap":
            continue
        if t_n == block["block"].shape[0] and t_m == block["block"].shape[1]:
            for mask_num, mask in enumerate(sample["train"][0]["masks"]):
                if time.time() - start_time > max_time:
                    break
                if t_n == mask["mask"].shape[0] and t_m == mask["mask"].shape[1]:
                    for color in range(10):
                        if (
                            target_image
                            == block["block"] * (1 - mask["mask"])
                            + mask["mask"] * color
                        ).all():
                            for color_dict in sample["train"][0]["colors"][
                                color
                            ].copy():
                                candidates.append(
                                    {
                                        "block": block["params"],
                                        "mask": {
                                            "params": mask["params"],
                                            "operation": mask["operation"],
                                        },
                                        "color": color_dict.copy(),
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
            status, block = get_predict(
                original_image,
                candidate["block"],
                sample["train"][k]["block_cache"],
                color_scheme=sample["train"][k],
            )
            if status != 0 or block.shape[0] != t_n or block.shape[1] != t_m:
                continue
            status, mask = get_mask_from_block_params(
                original_image,
                candidate["mask"],
                block_cache=sample["train"][k]["block_cache"],
                color_scheme=sample["train"][k],
                mask_cache=sample["train"][k]["mask_cache"],
            )
            if status != 0 or mask.shape[0] != t_n or mask.shape[1] != t_m:
                continue
            color = get_color(candidate["color"], sample["train"][k]["colors"])
            if color < 0:
                continue
            if (target_image == block * (1 - mask) + mask * color).all():
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
            status, block = get_predict(
                original_image,
                candidate["block"],
                color_scheme=color_scheme,
                block_cache=sample["train"][test_n]["block_cache"],
            )
            if status != 0:
                continue
            status, mask = get_mask_from_block_params(
                original_image,
                candidate["mask"],
                color_scheme=color_scheme,
                block_cache=sample["train"][test_n]["block_cache"],
                mask_cache=sample["train"][test_n]["mask_cache"],
            )
            if (
                status != 0
                or mask.shape[0] != block.shape[0]
                or mask.shape[1] != block.shape[1]
            ):
                continue
            color = get_color(candidate["color"], color_scheme["colors"])
            if color < 0:
                continue
            prediction = (block * (1 - mask)) + (mask * color)
            answers[test_n].append(np.rot90(prediction, k=-rotate_target))
            result_generated = True

    if result_generated:
        return 0, answers
    else:
        return 2, None


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


def generate_corners(
    original_image, simetry_type="rotate", block_size=None, color=None
):
    size = (original_image.shape[0] + 1) // 2, (original_image.shape[1] + 1) // 2
    # corners
    corners = []
    if simetry_type == "rotate":
        corners.append(original_image[: size[0], : size[1]])
        corners.append(np.rot90(original_image[: size[0], -size[1] :], 1))
        corners.append(np.rot90(original_image[-size[0] :, -size[1] :], 2))
        corners.append(np.rot90(original_image[-size[0] :, : size[1]], 3))
    elif simetry_type == "reflect":
        corners.append(original_image[: size[0], : size[1]])
        corners.append(original_image[: size[0], -size[1] :][:, ::-1])
        corners.append(original_image[-size[0] :, -size[1] :][::-1, ::-1])
        corners.append(original_image[-size[0] :, : size[1]][::-1, :])
        if original_image.shape[0] == original_image.shape[1]:
            mask = np.logical_and(original_image != color, original_image.T != color)
            if (original_image.T == original_image)[mask].all():
                corners.append(original_image[: size[0], : size[1]].T)
                corners.append(original_image[: size[0], -size[1] :][:, ::-1].T)
                corners.append(original_image[-size[0] :, -size[1] :][::-1, ::-1].T)
                corners.append(original_image[-size[0] :, : size[1]][::-1, :].T)

    elif simetry_type == "surface":
        for i in range(original_image.shape[0] // block_size[0]):
            for j in range(original_image.shape[1] // block_size[1]):
                corners.append(
                    original_image[
                        i * block_size[0] : (i + 1) * block_size[0],
                        j * block_size[1] : (j + 1) * block_size[1],
                    ]
                )

    return corners


def mosaic_reconstruction_check_corner_consistency(corners, color):
    for i, corner1 in enumerate(corners[:-1]):
        for corner2 in corners[i + 1 :]:
            mask = np.logical_and(corner1 != color, corner2 != color)
            if not (corner1 == corner2)[mask].all():
                return False
    return True


def mosaic_reconstruction_check_corner(
    original_image, target_image, color, simetry_types
):
    if not (original_image == target_image)[original_image != color].all():
        return False

    status, predicted_image = mosaic_reconstruct_corner(
        original_image, color, simetry_types
    )
    if status != 0:
        return False
    temp = (
        predicted_image[: original_image.shape[0], : original_image.shape[1]]
        == target_image
    )
    if (
        predicted_image[: original_image.shape[0], : original_image.shape[1]]
        == target_image
    ).all():
        return True
    return False


def mosaic_reconstruct_corner(original_image, color, simetry_types=None):
    # corners
    target_images = []
    extensions = []
    if simetry_types is None:
        simetry_types = ["rotate", "reflect", "surface"]

    for extensions_sum in range(20):
        for extension0 in range(extensions_sum):
            extension1 = extensions_sum - extension0
            for simetry_type in simetry_types:
                if simetry_type == "rotate" and (
                    original_image.shape[0] + extension0
                    != original_image.shape[1] + extension1
                ):
                    continue
                new_image = np.uint8(
                    np.ones(
                        (
                            original_image.shape[0] + extension0,
                            original_image.shape[1] + extension1,
                        )
                    )
                    * color
                )
                new_image[
                    : original_image.shape[0], : original_image.shape[1]
                ] = original_image

                if simetry_type in ["rotate", "reflect"]:
                    corners = generate_corners(new_image, simetry_type, color=color)
                    if not mosaic_reconstruction_check_corner_consistency(
                        corners, color
                    ):
                        continue
                elif simetry_type == "surface":
                    sizes_found = False
                    for block_sizes_sum in range(
                        2, min(15, new_image.shape[0] + new_image.shape[0] - 2)
                    ):
                        for block_size1 in range(
                            1, min(block_sizes_sum - 1, new_image.shape[0] - 1)
                        ):
                            if new_image.shape[0] % block_size1 != 0:
                                continue
                            block_size2 = min(
                                block_sizes_sum - block_size1, new_image.shape[1] - 1
                            )
                            if new_image.shape[1] % block_size2 != 0:
                                continue
                            corners = generate_corners(
                                new_image, simetry_type, (block_size1, block_size2)
                            )
                            if not mosaic_reconstruction_check_corner_consistency(
                                corners, color
                            ):
                                continue
                            else:
                                sizes_found = True
                                break
                        if sizes_found:
                            break
                    if not sizes_found:
                        continue

                final_corner = corners[0].copy()
                for i, corner in enumerate(corners[1:]):
                    mask = np.logical_and(final_corner == color, corner != color)
                    final_corner[mask] = corner[mask]
                if (final_corner == color).any() and final_corner.shape[
                    0
                ] == final_corner.shape[1]:
                    mask = final_corner == color
                    final_corner[mask] = final_corner.T[mask]

                size = final_corner.shape
                target_image = new_image.copy()
                target_image[: size[0], : size[1]] = final_corner
                if simetry_type == "rotate":
                    target_image[: size[0], -size[1] :] = np.rot90(final_corner, -1)
                    target_image[-size[0] :, -size[1] :] = np.rot90(final_corner, -2)
                    target_image[-size[0] :, : size[1]] = np.rot90(final_corner, -3)
                elif simetry_type == "reflect":
                    target_image[: size[0], -size[1] :] = final_corner[:, ::-1]
                    target_image[-size[0] :, -size[1] :] = final_corner[::-1, ::-1]
                    target_image[-size[0] :, : size[1]] = final_corner[::-1, :]
                elif simetry_type == "surface":
                    for i in range(new_image.shape[0] // size[0]):
                        for j in range(new_image.shape[1] // size[1]):
                            target_image[
                                i * size[0] : (i + 1) * size[0],
                                j * size[1] : (j + 1) * size[1],
                            ] = final_corner

                target_image = target_image[
                    : original_image.shape[0], : original_image.shape[1]
                ]
                extensions.append(extension0 + extension1)
                return 0, target_image

    return 1, None


def filter_list_of_dicts(list1, list2):
    final_list = []
    for item2 in list2:
        for item1 in list1:
            if get_dict_hash(item1) == get_dict_hash(item2):
                final_list.append(item1)
                break
    return final_list


def reflect_rotate_roll(
    image, reflect=(False, False), rotate=0, inverse=False, roll=(0, 0)
):
    if inverse:
        result = np.rot90(image, -rotate).copy()
    else:
        result = np.rot90(image, rotate).copy()
    if reflect[0]:
        result = result[::-1]
    if reflect[1]:
        result = result[:, ::-1]
    if inverse:
        result = np.roll(result, -roll[0], axis=0)
        result = np.roll(result, -roll[1], axis=1)
    else:
        result = np.roll(result, roll[0], axis=0)
        result = np.roll(result, roll[1], axis=1)

    return result


def mosaic_reconstruction(
    sample,
    rotate=0,
    simetry_types=None,
    reflect=(False, False),
    rotate_target=0,
    roll=(0, 0),
):
    color_candidates_final = []

    for k in range(len(sample["train"])):
        color_candidates = []
        original_image = np.rot90(np.uint8(sample["train"][k]["input"]), rotate)
        target_image = np.rot90(np.uint8(sample["train"][k]["output"]), rotate)
        target_image = reflect_rotate_roll(
            target_image,
            reflect=reflect,
            rotate=rotate_target,
            roll=roll,
            inverse=False,
        )
        if original_image.shape != target_image.shape:
            return 1, None
        for color_num in range(10):
            if mosaic_reconstruction_check_corner(
                original_image, target_image, color_num, simetry_types
            ):
                for color_dict in sample["train"][k]["colors"][color_num]:
                    color_candidates.append(color_dict)
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
        for color_dict in color_candidates_final:
            color = get_color(color_dict, color_scheme["colors"])
            status, prediction = mosaic_reconstruct_corner(
                np.rot90(original_image, rotate), color, simetry_types
            )
            if status != 0:
                continue
            prediction = reflect_rotate_roll(
                prediction,
                reflect=reflect,
                rotate=rotate_target,
                roll=roll,
                inverse=True,
            )
            answers[test_n].append(np.rot90(prediction, -rotate))
            result_generated = True

    if result_generated:
        return 0, answers
    else:
        return 3, None


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


def extract_mosaic_block(
    sample,
    rotate=0,
    simetry_types=None,
    reflect=(False, False),
    rotate_target=0,
    roll=(0, 0),
):
    color_candidates_final = []

    for k in range(len(sample["train"])):
        color_candidates = []
        original_image = np.rot90(np.uint8(sample["train"][k]["input"]), rotate)
        target_image = np.rot90(np.uint8(sample["train"][k]["output"]), rotate)
        target_image = reflect_rotate_roll(
            target_image,
            reflect=reflect,
            rotate=rotate_target,
            roll=roll,
            inverse=False,
        )
        for color_num in range(10):
            mask = original_image == color_num
            sum0 = mask.sum(0)
            sum1 = mask.sum(1)

            if len(np.unique(sum0)) != 2 or len(np.unique(sum1)) != 2:
                continue
            if target_image.shape[0] != max(sum0) or target_image.shape[1] != max(sum1):
                continue

            indices0 = np.arange(len(sum1))[sum1 > 0]
            indices1 = np.arange(len(sum0))[sum0 > 0]

            generated_target_image = original_image.copy()
            generated_target_image[
                indices0.min() : indices0.max() + 1, indices1.min() : indices1.max() + 1
            ] = target_image

            if mosaic_reconstruction_check_corner(
                original_image, generated_target_image, color_num, simetry_types
            ):
                for color_dict in sample["train"][k]["colors"][color_num]:
                    color_candidates.append(color_dict)
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
        original_image = np.rot90(np.uint8(test_data["input"]), rotate)
        color_scheme = get_color_scheme(original_image)
        for color_dict in color_candidates_final:
            color = get_color(color_dict, color_scheme["colors"])
            status, prediction = mosaic_reconstruct_corner(
                original_image, color, simetry_types
            )

            if status != 0:
                continue
            prediction = reflect_rotate_roll(
                prediction,
                reflect=reflect,
                rotate=rotate_target,
                roll=roll,
                inverse=True,
            )
            mask = original_image == color
            sum0 = mask.sum(0)
            sum1 = mask.sum(1)
            indices0 = np.arange(len(sum1))[sum1 > 0]
            indices1 = np.arange(len(sum0))[sum0 > 0]

            prediction = prediction[
                indices0.min() : indices0.max() + 1, indices1.min() : indices1.max() + 1
            ]

            answers[test_n].append(np.rot90(prediction, -rotate))
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
