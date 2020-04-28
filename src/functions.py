import numpy as np
from preprocessing import get_predict, find_grid, get_color, get_color_scheme


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
                candidates[n_factor][i].append(initial_values)
    return candidates


def filter_candidates(
    target_image,
    factors,
    intersection=0,
    candidates=None,
    original_image=None,
    blocks=None,
):
    t_n, t_m = target_image.shape
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
                        result, array = get_predict(original_image, data)
                        if result != 0:
                            break

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

        factors, new_candidates = filter_candidates(
            target_image,
            factors,
            intersection=intersection,
            candidates=candidates,
            original_image=original_image,
            blocks=None,
        )

    answers = []
    final_factor_n = -1

    # check if we have at least one valid solution
    for n_factor, factor in enumerate(factors):
        if factor[0] > 0 and factor[1] > 0:
            final_factor_n = n_factor
            break
    if final_factor_n == -1:
        return 1, None
    factor = factors[final_factor_n]

    for test_data in sample["test"]:
        original_image = np.array(test_data["input"])

        for i in range(factor[0]):
            for j in range(factor[1]):
                result, array = get_predict(
                    original_image, candidates[final_factor_n][i][j][0]
                )
                if result != 0:
                    return 5, None
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
                            grid_color_list[0],
                            get_color_scheme(original_image)["colors"],
                        )

                predict[
                    i * (n - intersection) : i * (n - intersection) + n,
                    j * (m - intersection) : j * (m - intersection) + m,
                ] = array

        answers.append(np.rot90(predict, k=-rotate_target))

    return 0, answers
