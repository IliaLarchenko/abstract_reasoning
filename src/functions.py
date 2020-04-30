import numpy as np
from src.preprocessing import (
    get_predict,
    find_grid,
    get_color,
    get_color_scheme,
    get_mask_from_block_params,
)


def initiate_candidates_list(factors, initial_values=None):
    """creating an empty candidates list corresponding to factors
    for each (m,n) factor it is m x n matrix of lists"""
    candidates = []
    if not initial_values:
        initial_values = []
    for n_factor, factor in enumerate(factors):
        candidates.append([])
        for i in range(factor[0]):
            candidates[n_factor].append([])
            for j in range(factor[1]):
                candidates[n_factor][i].append(initial_values.copy())
    return candidates


def filter_candidates(
    target_image,
    factors,
    intersection=0,
    candidates=None,
    original_image=None,
    blocks=None,
    blocks_cache=None,
):
    t_n, t_m = target_image.shape
    color_scheme = get_color_scheme(original_image)
    new_candidates = initiate_candidates_list(factors)
    for n_factor, factor in enumerate(factors.copy()):
        for i in range(factor[0]):
            for j in range(factor[1]):
                if blocks:
                    local_candidates = blocks
                else:
                    local_candidates = candidates[n_factor][i][j]

                for data in local_candidates:
                    if blocks:
                        array = data["block"]
                        params = data["params"]
                    else:
                        params = data
                        result, array = get_predict(
                            original_image, data, blocks_cache, color_scheme
                        )
                        if result != 0:
                            continue

                    n, m = array.shape

                    # work with valid candidates only
                    if n <= 0 or m <= 0:
                        continue
                    if (
                        n - intersection != (t_n - intersection) / factor[0]
                        or m - intersection != (t_m - intersection) / factor[1]
                    ):
                        continue

                    start_n = i * (n - intersection)
                    start_m = j * (m - intersection)

                    # checking the sizes of expected and proposed blocks
                    if not (
                        (
                            array.shape[0]
                            == target_image[
                                start_n : start_n + n, start_m : start_m + m
                            ].shape[0]
                        )
                        and (
                            array.shape[1]
                            == target_image[
                                start_n : start_n + n, start_m : j * start_m + m
                            ].shape[1]
                        )
                    ):
                        continue

                    # adding the candidate to the candidates list
                    if (
                        array
                        == target_image[start_n : start_n + n, start_m : start_m + m]
                    ).all():
                        new_candidates[n_factor][i][j].append(params)

                # if there is no candidates for one of the cells the whole factor is invalid
                if len(new_candidates[n_factor][i][j]) == 0:
                    factors[n_factor] = [0, 0]
                    break
            if factors[n_factor][0] == 0:
                break

    return factors, new_candidates


def mosaic(sample, rotate_target=0, intersection=0):
    """ combines all possible combinations of blocks into target image"""
    target_image = np.rot90(np.array(sample["train"][0]["output"]), rotate_target)
    t_n, t_m = target_image.shape
    factors = []

    if intersection < 0:
        grid_color, grid_size = find_grid(target_image)
        if grid_color < 0:
            return 5, None
        factors = [grid_size]
        grid_color_list = sample["processed_train"][0]["colors"][grid_color]
    else:
        for i in range(1, t_n):
            for j in range(1, t_m):
                if (t_n - intersection) % i == 0 and (t_m - intersection) % j == 0:
                    factors.append([i, j])

    # get the initial candidates
    factors, candidates = filter_candidates(
        target_image,
        factors,
        intersection=intersection,
        candidates=None,
        original_image=None,
        blocks=sample["processed_train"][0]["blocks"],
    )

    # filter them, leave only those that ok for all train samples
    for k in range(1, len(sample["train"])):
        if intersection < 0:
            grid_color, grid_size = find_grid(target_image)
            if grid_color < 0:
                return 5, None
            if (factors[0][0] != grid_size[0]) or (factors[0][1] != grid_size[1]):
                return 6, None
            new_grid_color_list = []
            for color_dict in grid_color_list:
                if (
                    get_color(color_dict, sample["processed_train"][k]["colors"])
                    == grid_color
                ):
                    new_grid_color_list.append(color_dict)
            if len(new_grid_color_list) == 0:
                return 7, None
            else:
                grid_color_list = new_grid_color_list.copy()

        original_image = np.array(sample["train"][k]["input"])
        target_image = np.rot90(np.array(sample["train"][k]["output"]), rotate_target)
        if "block_cache" not in sample["processed_train"][k]:
            sample["processed_train"][k]["block_cache"] = {}

        factors, candidates = filter_candidates(
            target_image,
            factors,
            intersection=intersection,
            candidates=candidates,
            original_image=original_image,
            blocks=None,
            blocks_cache=sample["processed_train"][k]["block_cache"],
        )

    answers = []
    for _ in sample["test"]:
        answers.append([])
    final_factor_n = -1

    # check if we have at least one valid solution
    for n_factor, factor in enumerate(factors):
        if factor[0] > 0 and factor[1] > 0:
            final_factor_n = n_factor
            factor = factors[final_factor_n]

            for test_n, test_data in enumerate(sample["test"]):
                original_image = np.array(test_data["input"])
                color_scheme = get_color_scheme(original_image)
                skip = False
                for i in range(factor[0]):
                    for j in range(factor[1]):
                        result, array = get_predict(
                            original_image,
                            candidates[final_factor_n][i][j][0],
                            color_scheme,
                        )
                        if result != 0:
                            skip = True
                            break
                        n, m = array.shape
                        if i == 0 and j == 0:
                            predict = np.int32(
                                np.zeros(
                                    (
                                        (n - intersection) * factor[0] + intersection,
                                        (m - intersection) * factor[1] + intersection,
                                    )
                                )
                            )
                            if intersection < 0:
                                predict += get_color(
                                    grid_color_list[0], color_scheme["colors"]
                                )

                        predict[
                            i * (n - intersection) : i * (n - intersection) + n,
                            j * (m - intersection) : j * (m - intersection) + m,
                        ] = array
                    if skip:
                        break
                if not skip:
                    answers[test_n].append(np.rot90(predict, k=-rotate_target))

    if final_factor_n == -1:
        return 1, None

    return 0, answers


def mask_to_blocks(sample, rotate_target=0):
    target_image = np.rot90(np.array(sample["train"][0]["output"]), rotate_target)
    t_n, t_m = target_image.shape
    candidates = []
    for block in sample["processed_train"][0]["blocks"]:
        if t_n == block["block"].shape[0] and t_m == block["block"].shape[1]:
            for mask in sample["processed_train"][0]["masks"]:
                if t_n == mask["mask"].shape[0] and t_m == mask["mask"].shape[1]:
                    for color in range(10):
                        if (
                            target_image
                            == block["block"] * (1 - mask["mask"])
                            + mask["mask"] * color
                        ).all():
                            for color_dict in sample["processed_train"][0]["colors"][
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

    for k in range(1, len(sample["train"])):
        original_image = np.array(sample["train"][k]["input"])
        target_image = np.rot90(np.array(sample["train"][k]["output"]), rotate_target)
        t_n, t_m = target_image.shape
        new_candidates = []

        if "block_cache" not in sample["processed_train"][k]:
            sample["processed_train"][k]["block_cache"] = {}
        if "mask_cache" not in sample["processed_train"][k]:
            sample["processed_train"][k]["mask_cache"] = {}

        for candidate in candidates:
            status, block = get_predict(
                original_image,
                candidate["block"],
                sample["processed_train"][k]["block_cache"],
                color_scheme=sample["processed_train"][k],
            )
            if status != 0 or block.shape[0] != t_n or block.shape[1] != t_m:
                continue
            status, mask = get_mask_from_block_params(
                original_image,
                candidate["mask"],
                block_cache=sample["processed_train"][k]["block_cache"],
                color_scheme=sample["processed_train"][k],
                mask_cache=sample["processed_train"][k]["mask_cache"],
            )
            if status != 0 or mask.shape[0] != t_n or mask.shape[1] != t_m:
                continue
            color = get_color(
                candidate["color"], sample["processed_train"][k]["colors"]
            )
            if color < 0:
                continue
            if (target_image == block * (1 - mask) + mask * color).all():
                new_candidates.append(candidate)
        candidates = new_candidates.copy()

    if len(candidates) == 0:
        return 1, None

    answers = []
    for _ in sample["test"]:
        answers.append([])

    result_generated = False
    for test_n, test_data in enumerate(sample["test"]):
        original_image = np.array(test_data["input"])
        color_scheme = get_color_scheme(original_image)
        for candidate in candidates:
            status, block = get_predict(
                original_image, candidate["block"], color_scheme=color_scheme
            )
            if status != 0:
                continue
            status, mask = get_mask_from_block_params(
                original_image, candidate["mask"], color_scheme=color_scheme
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
            prediction = block * (1 - mask) + mask * color
            answers[test_n].append(np.rot90(prediction, k=-rotate_target))
            result_generated = True

    if result_generated:
        return 0, answers
    else:
        return 2, None


def paint_mask(sample, rotate_target=0):
    target_image = np.rot90(np.array(sample["train"][0]["output"]), rotate_target)
    unique = np.unique(target_image)
    if len(unique) > 2:
        return 3, None
    t_n, t_m = target_image.shape
    candidates = []
    for mask in sample["processed_train"][0]["masks"]:
        if t_n == mask["mask"].shape[0] and t_m == mask["mask"].shape[1]:
            unique = np.unique(target_image[mask["mask"]])
            if len(unique) != 1:
                continue
            color2 = unique[0]
            unique = np.unique(target_image[np.logical_not(mask["mask"])])
            if len(unique) != 1:
                continue
            color1 = unique[0]
            for color_dict1 in sample["processed_train"][0]["colors"][color1].copy():
                for color_dict2 in sample["processed_train"][0]["colors"][
                    color2
                ].copy():
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

    for k in range(1, len(sample["train"])):
        original_image = np.array(sample["train"][k]["input"])
        target_image = np.rot90(np.array(sample["train"][k]["output"]), rotate_target)
        t_n, t_m = target_image.shape
        new_candidates = []
        if "block_cache" not in sample["processed_train"][k]:
            sample["processed_train"][k]["block_cache"] = {}
        if "mask_cache" not in sample["processed_train"][k]:
            sample["processed_train"][k]["mask_cache"] = {}

        for candidate in candidates:
            status, mask = get_mask_from_block_params(
                original_image,
                candidate["mask"],
                block_cache=sample["processed_train"][k]["block_cache"],
                color_scheme=sample["processed_train"][k],
                mask_cache=sample["processed_train"][k]["mask_cache"],
            )
            if status != 0 or mask.shape[0] != t_n or mask.shape[1] != t_m:
                continue
            color1 = get_color(
                candidate["color1"], sample["processed_train"][k]["colors"]
            )
            color2 = get_color(
                candidate["color2"], sample["processed_train"][k]["colors"]
            )

            if color1 < 0 or color2 < 0:
                continue

            part1 = (1 - mask) * color1
            part2 = mask * color2
            result = part1 + part2
            result == target_image
            if (target_image == ((1 - mask) * color1 + mask * color2)).all():
                new_candidates.append(candidate)
        candidates = new_candidates.copy()

    if len(candidates) == 0:
        return 1, None

    answers = []
    for _ in sample["test"]:
        answers.append([])

    result_generated = False
    for test_n, test_data in enumerate(sample["test"]):
        original_image = np.array(test_data["input"])
        color_scheme = get_color_scheme(original_image)
        for candidate in candidates:
            status, mask = get_mask_from_block_params(
                original_image, candidate["mask"], color_scheme=color_scheme
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
