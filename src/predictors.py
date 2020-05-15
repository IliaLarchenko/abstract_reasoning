import numpy as np
from src.preprocessing import get_color, get_color_scheme, get_dict_hash
from src.functions import (
    filter_list_of_dicts,
    combine_two_lists,
    intersect_two_lists,
    swap_two_colors,
)
from src.preprocessing import find_grid, get_predict, get_mask_from_block_params
import random
from scipy import ndimage

import itertools


class predictor:
    def __init__(self, params=None, preprocess_params=None):
        self.params = params
        self.preprocess_params = preprocess_params
        self.solution_candidates = []
        if self.params is not None and "rrr_input" in params:
            self.rrr_input = params["rrr_input"]
        else:
            self.rrr_input = True

    def retrive_params_values(self, params, color_scheme):
        new_params = {}
        for k, v in params.items():
            if k[-5:] == "color":
                new_params[k] = get_color(v, color_scheme["colors"])
                if new_params[k] < 0:
                    return 1, None
            else:
                new_params[k] = v
        return 0, new_params

    def reflect_rotate_roll(self, image, inverse=False):
        if self.params is not None and "reflect" in self.params:
            reflect = self.params["reflect"]
        else:
            reflect = (False, False)
        if self.params is not None and "rotate" in self.params:
            rotate = self.params["rotate"]
        else:
            rotate = 0
        if self.params is not None and "roll" in self.params:
            roll = self.params["roll"]
        else:
            roll = (0, 0)

        result = image.copy()

        if inverse:
            if reflect[0]:
                result = result[::-1]
            if reflect[1]:
                result = result[:, ::-1]
            result = np.rot90(result, -rotate)
            result = np.roll(result, -roll[1], axis=1)
            result = np.roll(result, -roll[0], axis=0)
        else:
            result = np.roll(result, roll[0], axis=0)
            result = np.roll(result, roll[1], axis=1)
            result = np.rot90(result, rotate)
            if reflect[1]:
                result = result[:, ::-1]
            if reflect[0]:
                result = result[::-1]

        return result

    def get_images(self, k, train=True):
        if train:
            if self.rrr_input:
                original_image = self.reflect_rotate_roll(
                    np.uint8(self.sample["train"][k]["input"])
                )
            else:
                original_image = np.uint8(self.sample["train"][k]["input"])
            target_image = self.reflect_rotate_roll(
                np.uint8(self.sample["train"][k]["output"])
            )
            return original_image, target_image
        else:
            if self.rrr_input:
                original_image = self.reflect_rotate_roll(
                    np.uint8(self.sample["test"][k]["input"])
                )
            else:
                original_image = np.uint8(self.sample["test"][k]["input"])
            return original_image

    def process_prediction(self, image):
        return self.reflect_rotate_roll(image, inverse=True)

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        return 1, None

    def filter_colors(self):
        for i in range(10):
            list_of_colors = [
                get_dict_hash(color_dict)
                for color_dict in self.sample["train"][0]["colors"][i]
            ]
            for color_scheme in self.sample["train"][1:]:
                new_set = set(
                    [
                        get_dict_hash(color_dict)
                        for color_dict in color_scheme["colors"][i]
                    ]
                )
                list_of_colors = [x for x in list_of_colors if x in new_set]
                if len(list_of_colors) == 0:
                    break
            if len(list_of_colors) > 1:
                colors_to_delete = list_of_colors[1:]

                for color_scheme in self.sample["train"]:
                    for color_dict in color_scheme["colors"][i].copy():
                        if get_dict_hash(color_dict) in colors_to_delete:
                            color_scheme["colors"][i].remove(color_dict)
        return

    def init_call(self):
        self.filter_colors()

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)
        return 0

    def process_full_train(self):
        for k in range(len(self.sample["train"])):
            status = self.process_one_sample(k, initial=(k == 0))
            if status != 0:
                return 1

        if len(self.solution_candidates) == 0:
            return 2

        return 0

    def add_candidates_list(self, image, target_image, color_scheme, params):
        old_params = params.copy()
        params = params.copy()
        params["color_scheme"] = color_scheme
        params["block_cache"] = color_scheme["blocks"]
        params["mask_cache"] = color_scheme["masks"]

        status, prediction = self.predict_output(image, params)
        if (
            status != 0
            or prediction.shape != target_image.shape
            or not (prediction == target_image).all()
        ):
            return []

        result = [old_params.copy()]
        for k, v in params.copy().items():
            if k[-5:] == "color":
                temp_result = result.copy()
                result = []
                for dict in temp_result:
                    for color_dict in color_scheme["colors"][v]:
                        temp_dict = dict.copy()
                        temp_dict[k] = color_dict
                        result.append(temp_dict)

        return result

    def update_solution_candidates(self, local_candidates, initial):
        if initial:
            self.solution_candidates = local_candidates
        else:
            self.solution_candidates = filter_list_of_dicts(
                local_candidates, self.solution_candidates
            )
        if len(self.solution_candidates) == 0:
            return 4
        else:
            return 0

    def __call__(self, sample):
        """ works like fit_predict"""
        self.sample = sample
        self.init_call()
        self.initial_train = list(sample["train"]).copy()

        if self.params is not None and "skip_train" in self.params:
            skip_train = min(len(sample["train"]) - 2, self.params["skip_train"])
            train_len = len(self.initial_train) - skip_train
        else:
            train_len = len(self.initial_train)

        answers = []
        for _ in self.sample["test"]:
            answers.append([])
        result_generated = False

        all_subsets = list(itertools.combinations(self.initial_train, train_len))
        for subset in all_subsets:
            self.sample["train"] = subset
            status = self.process_full_train()
            if status != 0:
                continue

            for test_n, test_data in enumerate(self.sample["test"]):
                original_image = self.get_images(test_n, train=False)
                color_scheme = self.sample["test"][test_n]
                for params_dict in self.solution_candidates:
                    status, params = self.retrive_params_values(
                        params_dict, color_scheme
                    )
                    if status != 0:
                        continue
                    params["block_cache"] = self.sample["test"][test_n]["blocks"]
                    params["mask_cache"] = self.sample["test"][test_n]["masks"]
                    params["color_scheme"] = self.sample["test"][test_n]
                    status, prediction = self.predict_output(original_image, params)
                    if status != 0:
                        continue

                    answers[test_n].append(self.process_prediction(prediction))
                    result_generated = True

        self.sample["train"] = self.initial_train
        if result_generated:
            return 0, answers
        else:
            return 3, None


class fill(predictor):
    """inner fills all pixels around all pixels with particular color with new color
    outer fills the pixels with fill color if all neighbour colors have background color"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)
        # self.type = params["type"]  # inner or outer
        if "pattern" in params:
            self.pattern = params["pattern"]
        else:
            self.pattern = np.array(
                [[True, True, True], [True, False, True], [True, True, True]]
            )

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        result = image.copy()
        if (
            params["process_type"] == "isolated"
            or params["process_type"] == "isolated_non_bg"
        ):
            image_with_borders = (
                np.ones((image.shape[0] + 2, image.shape[1] + 2))
                * params["background_color"]
            )
        else:
            image_with_borders = np.ones((image.shape[0] + 2, image.shape[1] + 2)) * 11
        image_with_borders[1:-1, 1:-1] = image
        for i in range(1, image_with_borders.shape[0] - 1):
            for j in range(1, image_with_borders.shape[1] - 1):
                if params["process_type"] == "outer":
                    if image[i - 1, j - 1] == params["fill_color"]:
                        image_with_borders[i - 1 : i + 2, j - 1 : j + 2][
                            np.array(self.pattern)
                        ] = params["background_color"]
                elif params["process_type"] == "inner":
                    if (
                        image_with_borders[i - 1 : i + 2, j - 1 : j + 2][
                            np.array(self.pattern)
                        ]
                        == params["background_color"]
                    ).all():
                        result[i - 1, j - 1] = params["fill_color"]
                elif params["process_type"] == "inner_ignore_background":
                    if (
                        image_with_borders[i - 1 : i + 2, j - 1 : j + 2][
                            np.array(self.pattern)
                        ]
                        != params["background_color"]
                    ).all():
                        result[i - 1, j - 1] = params["fill_color"]
                elif params["process_type"] == "isolated":
                    if not (
                        image_with_borders[i - 1 : i + 2, j - 1 : j + 2][
                            np.array(self.pattern)
                        ]
                        == params["fill_color"]
                    ).any():
                        result[i - 1, j - 1] = params["background_color"]
                elif params["process_type"] == "isolated_non_bg":
                    if (
                        image_with_borders[i - 1 : i + 2, j - 1 : j + 2][
                            np.array(self.pattern)
                        ]
                        == params["background_color"]
                    ).all() and image[i - 1, j - 1] != params["background_color"]:
                        result[i - 1, j - 1] = params["fill_color"]
                elif params["process_type"] == "full":
                    if (
                        i - 1 + self.pattern.shape[0] > image.shape[0]
                        or j - 1 + self.pattern.shape[1] > image.shape[1]
                    ):
                        continue
                    if (
                        image[
                            i - 1 : i - 1 + self.pattern.shape[0],
                            j - 1 : j - 1 + self.pattern.shape[1],
                        ][np.array(self.pattern)]
                        == params["background_color"]
                    ).all():
                        result[
                            i - 1 : i - 1 + self.pattern.shape[0],
                            j - 1 : j - 1 + self.pattern.shape[1],
                        ][np.array(self.pattern)] = params["fill_color"]

                else:
                    return 6, None
        if params["process_type"] == "outer":
            result = image_with_borders[1:-1, 1:-1]
        return 0, result

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)
        if original_image.shape != target_image.shape:
            return 5, None
        for background_color in range(10):
            if not (target_image == background_color).any():
                continue
            for fill_color in range(10):
                if not (target_image == fill_color).any():
                    continue
                mask = np.logical_and(
                    target_image != background_color, target_image != fill_color
                )
                if not (target_image == original_image)[mask].all():
                    continue
                for process_type in [
                    "outer",
                    "full",
                    "isolated_non_bg",
                    "isolated",
                    "inner_ignore_background",
                    "inner",
                ]:
                    params = {
                        "background_color": background_color,
                        "fill_color": fill_color,
                        "process_type": process_type,
                    }

                    local_candidates = local_candidates + self.add_candidates_list(
                        original_image, target_image, self.sample["train"][k], params
                    )
        return self.update_solution_candidates(local_candidates, initial)


class puzzle(predictor):
    """inner fills all pixels around all pixels with particular color with new color
    outer fills the pixels with fill color if all neighbour colors have background color"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)
        self.intersection = params["intersection"]

    def initiate_factors(self, target_image):
        t_n, t_m = target_image.shape
        factors = []
        grid_color_list = []
        if self.intersection < 0:
            grid_color, grid_size = find_grid(target_image)
            if grid_color < 0:
                return factors, []
            factors = [grid_size]
            grid_color_list = self.sample["train"][0]["colors"][grid_color]
        else:
            for i in range(1, t_n + 1):
                for j in range(1, t_m + 1):
                    if (t_n - self.intersection) % i == 0 and (
                        t_m - self.intersection
                    ) % j == 0:
                        factors.append([i, j])
        return factors, grid_color_list

    def retrive_params_values(self, params, color_scheme):
        pass

    def predict_output(self, image, color_scheme, factor, params, block_cache):
        """ predicts 1 output image given input image and prediction params"""
        skip = False
        for i in range(factor[0]):
            for j in range(factor[1]):
                status, array = get_predict(
                    image, params[i][j][0], block_cache, color_scheme
                )
                if status != 0:
                    skip = True
                    break

                if i == 0 and j == 0:
                    n, m = array.shape
                    predict = np.uint8(
                        np.zeros(
                            (
                                (n - self.intersection) * factor[0] + self.intersection,
                                (m - self.intersection) * factor[1] + self.intersection,
                            )
                        )
                    )
                    if self.intersection < 0:
                        predict += get_color(
                            self.grid_color_list[0], color_scheme["colors"]
                        )
                else:
                    if n != array.shape[0] or m != array.shape[1]:
                        skip = True
                        break

                predict[
                    i * (n - self.intersection) : i * (n - self.intersection) + n,
                    j * (m - self.intersection) : j * (m - self.intersection) + m,
                ] = array
            if skip:
                return 1, None

        return 0, predict

    def initiate_candidates_list(self, initial_values=None):
        """creates an empty candidates list corresponding to factors
        for each (m,n) factor it is m x n matrix of lists"""
        candidates = []
        if not initial_values:
            initial_values = []
        for n_factor, factor in enumerate(self.factors):
            candidates.append([])
            for i in range(factor[0]):
                candidates[n_factor].append([])
                for j in range(factor[1]):
                    candidates[n_factor][i].append(initial_values.copy())
        return candidates

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""

        original_image, target_image = self.get_images(k)

        candidates_num = 0
        t_n, t_m = target_image.shape
        color_scheme = self.sample["train"][k]
        new_candidates = self.initiate_candidates_list()
        for n_factor, factor in enumerate(self.factors.copy()):
            for i in range(factor[0]):
                for j in range(factor[1]):
                    if initial:
                        local_candidates = self.sample["train"][k]["blocks"][
                            "arrays"
                        ].keys()
                        # print(local_candidates)
                    else:
                        local_candidates = self.solution_candidates[n_factor][i][j]

                    for data in local_candidates:
                        if initial:
                            # print(data)
                            array = self.sample["train"][k]["blocks"]["arrays"][data][
                                "array"
                            ]
                            params = self.sample["train"][k]["blocks"]["arrays"][data][
                                "params"
                            ]
                        else:
                            params = [data]
                            status, array = get_predict(
                                original_image,
                                data,
                                self.sample["train"][k]["blocks"],
                                color_scheme,
                            )
                            if status != 0:
                                continue

                        n, m = array.shape
                        # work with valid candidates only
                        if n <= 0 or m <= 0:
                            continue
                        if (
                            n - self.intersection
                            != (t_n - self.intersection) / factor[0]
                            or m - self.intersection
                            != (t_m - self.intersection) / factor[1]
                        ):
                            continue

                        start_n = i * (n - self.intersection)
                        start_m = j * (m - self.intersection)

                        if not (
                            (
                                n
                                == target_image[
                                    start_n : start_n + n, start_m : start_m + m
                                ].shape[0]
                            )
                            and (
                                m
                                == target_image[
                                    start_n : start_n + n, start_m : start_m + m
                                ].shape[1]
                            )
                        ):
                            continue

                        # adding the candidate to the candidates list
                        if (
                            array
                            == target_image[
                                start_n : start_n + n, start_m : start_m + m
                            ]
                        ).all():
                            new_candidates[n_factor][i][j].extend(params)
                            candidates_num += 1
                    # if there is no candidates for one of the cells the whole factor is invalid
                    if len(new_candidates[n_factor][i][j]) == 0:
                        self.factors[n_factor] = [0, 0]
                        break
                if self.factors[n_factor][0] == 0:
                    break

        self.solution_candidates = new_candidates

        if candidates_num > 0:
            return 0
        else:
            return 1

    def filter_factors(self, local_factors):
        for factor in self.factors:
            found = False
            for new_factor in local_factors:
                if factor == new_factor:
                    found = True
                    break
            if not found:
                factor = [0, 0]

        return

    def process_full_train(self):

        for k in range(len(self.sample["train"])):
            original_image, target_image = self.get_images(k)
            if k == 0:
                self.factors, self.grid_color_list = self.initiate_factors(target_image)
            else:
                local_factors, grid_color_list = self.initiate_factors(target_image)
                self.filter_factors(local_factors)
                self.grid_color_list = filter_list_of_dicts(
                    grid_color_list, self.grid_color_list
                )

            status = self.process_one_sample(k, initial=(k == 0))
            if status != 0:
                return 1

        if len(self.solution_candidates) == 0:
            return 2

        return 0

    def __call__(self, sample):
        """ works like fit_predict"""
        self.sample = sample
        status = self.process_full_train()
        if status != 0:
            return status, None

        answers = []
        for _ in self.sample["test"]:
            answers.append([])

        result_generated = False
        for test_n, test_data in enumerate(self.sample["test"]):
            original_image = self.get_images(test_n, train=False)
            color_scheme = self.sample["test"][test_n]
            for n_factor, factor in enumerate(self.factors):
                if factor[0] > 0 and factor[1] > 0:
                    status, prediction = self.predict_output(
                        original_image,
                        color_scheme,
                        factor,
                        self.solution_candidates[n_factor],
                        self.sample["test"][test_n]["blocks"],
                    )
                    if status == 0:
                        answers[test_n].append(self.process_prediction(prediction))
                        result_generated = True

        if result_generated:
            return 0, answers
        else:
            return 3, None


class pattern(predictor):
    """applies pattern to every pixel with particular color"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)
        # self.type = params["type"]

    def get_patterns(self, original_image, target_image):
        pattern_list = []
        if target_image.shape[0] % original_image.shape[0] != 0:
            self.try_self = False
            return []
        if target_image.shape[1] % original_image.shape[1] != 0:
            self.try_self = False
            return []

        size = [
            target_image.shape[0] // original_image.shape[0],
            target_image.shape[1] // original_image.shape[1],
        ]

        if size[0] != original_image.shape[0] or size[1] != original_image.shape[1]:
            self.try_self = False

        if max(size) == 1:
            return []
        for i in range(original_image.shape[0]):
            for j in range(original_image.shape[1]):
                current_block = target_image[
                    i * size[0] : (i + 1) * size[0], j * size[1] : (j + 1) * size[1]
                ]
                pattern_list = combine_two_lists(pattern_list, [current_block])

        return pattern_list

    def init_call(self):
        self.try_self = True
        for k in range(len(self.sample["train"])):
            original_image, target_image = self.get_images(k)
            patterns = self.get_patterns(original_image, target_image)
            if k == 0:
                self.all_patterns = patterns
            else:
                self.all_patterns = intersect_two_lists(self.all_patterns, patterns)
        if self.try_self:
            self.additional_patterns = ["self", "processed"]
        else:
            self.additional_patterns = []

    def predict_output(self, image, params):
        if params["swap"]:
            status, new_image = swap_two_colors(image)
            if status != 0:
                new_image = image
        else:
            new_image = image
        mask = new_image == params["mask_color"]
        if params["pattern_num"] == "self":
            pattern = image
        elif params["pattern_num"] == "processed":
            pattern = new_image
        else:
            pattern = self.all_patterns[params["pattern_num"]]

        size = (mask.shape[0] * pattern.shape[0], mask.shape[1] * pattern.shape[1])
        result = np.ones(size) * params["background_color"]
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] != params["inverse"]:
                    result[
                        i * pattern.shape[0] : (i + 1) * pattern.shape[0],
                        j * pattern.shape[1] : (j + 1) * pattern.shape[1],
                    ] = pattern

        return 0, result

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        if len(self.all_patterns) + len(self.additional_patterns) == 0:
            return 6

        for pattern_num in (
            list(range(len(self.all_patterns))) + self.additional_patterns
        ):
            for mask_color in range(10):
                if not (original_image == mask_color).any():
                    continue
                for background_color in range(10):
                    if not (target_image == background_color).any():
                        continue
                    for inverse in [True, False]:
                        for swap in [True, False]:
                            params = {
                                "pattern_num": pattern_num,
                                "mask_color": mask_color,
                                "background_color": background_color,
                                "inverse": inverse,
                                "swap": swap,
                            }

                            status, predict = self.predict_output(
                                original_image, params
                            )

                            if status == 0 and (predict == target_image).all():
                                local_candidates = (
                                    local_candidates
                                    + self.add_candidates_list(
                                        original_image,
                                        target_image,
                                        self.sample["train"][k],
                                        params,
                                    )
                                )

        return self.update_solution_candidates(local_candidates, initial)


class mask_to_block(predictor):
    """applies several masks to block"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)
        if params is not None and "mask_num" in params:
            self.mask_num = params["mask_num"]
        else:
            self.mask_num = 1

    def apply_mask(self, image, mask, color):
        if image.shape != mask.shape:
            return 1, None
        result = image.copy()
        result[mask] = color
        return 0, result

    def predict_output(self, image, params):
        status, block = get_predict(
            image,
            params["block"],
            block_cache=params["block_cache"],
            color_scheme=params["color_scheme"],
        )

        if status != 0:
            return status, None
        result = block

        for mask_param, color_param in zip(params["masks"], params["colors"]):
            status, mask = get_mask_from_block_params(
                image,
                mask_param,
                block_cache=params["block_cache"],
                mask_cache=params["mask_cache"],
                color_scheme=params["color_scheme"],
            )
            if status != 0:
                return status, None
            color = get_color(color_param, params["color_scheme"]["colors"])
            if color < 0:
                return 6, None
            status, result = self.apply_mask(result, mask, color)
            if status != 0:
                return status, None

        return 0, result

    def find_mask_color(self, target, mask, ignore_mask):
        visible_mask = np.logical_and(np.logical_not(ignore_mask), mask)
        if not (visible_mask).any():
            return -1
        visible_part = target[visible_mask]
        colors = np.unique(visible_part)
        if len(colors) == 1:
            return colors[0]
        else:
            return -1

    def add_block(self, target_image, ignore_mask, k):
        results = []
        for block_hash, block in self.sample["train"][k]["blocks"]["arrays"].items():
            # print(ignore_mask)
            if (block["array"].shape == target_image.shape) and (
                block["array"][np.logical_not(ignore_mask)]
                == target_image[np.logical_not(ignore_mask)]
            ).all():
                results.append(block_hash)

        if len(results) == 0:
            return 1, None
        else:
            return 0, results

    def generate_result(self, target_image, masks, colors, ignore_mask, k):
        if len(masks) == self.mask_num:
            status, blocks = self.add_block(target_image, ignore_mask, k)
            if status != 0:
                return 8, None
            result = [
                {"block": block, "masks": masks, "colors": colors} for block in blocks
            ]
            return 0, result

        result = []
        for mask_hash, mask in self.sample["train"][k]["masks"]["arrays"].items():
            if mask_hash in masks:
                continue
            if mask["array"].shape != target_image.shape:
                continue
            color = self.find_mask_color(target_image, mask["array"], ignore_mask)
            if color < 0:
                continue
            new_ignore_mask = np.logical_or(mask["array"], ignore_mask)
            status, new_results = self.generate_result(
                target_image, [mask_hash] + masks, [color] + colors, new_ignore_mask, k
            )
            if status != 0:
                continue
            result = result + new_results

        if len(result) <= 0:
            return 9, None
        else:
            return 0, result

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""

        candidates = []
        original_image, target_image = self.get_images(k)

        if initial:
            ignore_mask = np.zeros_like(target_image, dtype=bool)
            status, candidates = self.generate_result(
                target_image, [], [], ignore_mask, k
            )
            if status != 0:
                return status
            candidates = [
                {"block": block_params, "masks": x["masks"], "colors": x["colors"]}
                for x in candidates
                for block_params in self.sample["train"][k]["blocks"]["arrays"][
                    x["block"]
                ]["params"]
            ]
            for i in range(self.mask_num):
                candidates = [
                    {
                        "block": x["block"],
                        "masks": [
                            x["masks"][j] if j != i else mask_param
                            for j in range(self.mask_num)
                        ],
                        "colors": [
                            x["colors"][j] if j != i else color_param
                            for j in range(self.mask_num)
                        ],
                    }
                    for x in candidates
                    for mask_param in self.sample["train"][k]["masks"]["arrays"][
                        x["masks"][i]
                    ]["params"]
                    for color_param in self.sample["train"][k]["colors"][x["colors"][i]]
                ]

        else:
            for candidate in self.solution_candidates:
                params = candidate.copy()
                params["block_cache"] = self.sample["train"][k]["blocks"]
                params["mask_cache"] = self.sample["train"][k]["masks"]
                params["color_scheme"] = self.sample["train"][k]

                status, prediction = self.predict_output(original_image, params)
                if status != 0:
                    continue
                if (
                    prediction.shape == target_image.shape
                    and (prediction == target_image).all()
                ):
                    candidates.append(candidate)

        self.solution_candidates = candidates
        if len(self.solution_candidates) == 0:
            return 10

        return 0

    def __call__(self, sample):
        """ works like fit_predict"""
        self.sample = sample
        self.init_call()
        self.initial_train = list(sample["train"]).copy()

        if self.params is not None and "skip_train" in self.params:
            skip_train = min(len(sample["train"]) - 2, self.params["skip_train"])
            train_len = len(self.initial_train) - skip_train
        else:
            train_len = len(self.initial_train)

        answers = []
        for _ in self.sample["test"]:
            answers.append([])
        result_generated = False

        all_subsets = list(itertools.combinations(self.initial_train, train_len))
        for subset in all_subsets:
            self.sample["train"] = subset
            status = self.process_full_train()
            if status != 0:
                return status, None

            random.shuffle(self.solution_candidates)
            self.solution_candidates = self.solution_candidates[:300]
            print(len(self.solution_candidates))
            for test_n, test_data in enumerate(self.sample["test"]):
                original_image = self.get_images(test_n, train=False)
                color_scheme = self.sample["test"][test_n]
                for params_dict in self.solution_candidates:
                    params = params_dict.copy()
                    params["block_cache"] = self.sample["test"][test_n]["blocks"]
                    params["mask_cache"] = self.sample["test"][test_n]["masks"]
                    params["color_scheme"] = color_scheme

                    status, prediction = self.predict_output(original_image, params)
                    if status != 0:
                        continue

                    answers[test_n].append(self.process_prediction(prediction))
                    result_generated = True

        if result_generated:
            return 0, answers
        else:
            return 3, None


class pattern_from_blocks(pattern):
    """applies pattern to every pixel with particular color"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_output(self, image, params, pattern=None, mask=None, target_image=None):
        if pattern is None:
            status, pattern = get_predict(
                image,
                params["pattern"],
                block_cache=params["block_cache"],
                color_scheme=params["color_scheme"],
            )
            if status != 0:
                return 1, None
        if mask is None:
            status, mask = get_mask_from_block_params(
                image,
                params["mask"],
                block_cache=params["block_cache"],
                mask_cache=params["mask_cache"],
                color_scheme=params["color_scheme"],
            )
            if status != 0:
                return 2, None
        if target_image is not None:
            big_mask = np.repeat(
                np.repeat(mask, pattern.shape[0], 0), pattern.shape[1], 1
            )
            if not (
                target_image[np.logical_not(big_mask)] == params["background_color"]
            ).all():
                return 7, None

        size = (mask.shape[0] * pattern.shape[0], mask.shape[1] * pattern.shape[1])
        result = np.ones(size) * params["background_color"]

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    if (
                        target_image is not None
                        and not (
                            target_image[
                                i * pattern.shape[0] : (i + 1) * pattern.shape[0],
                                j * pattern.shape[1] : (j + 1) * pattern.shape[1],
                            ]
                            == pattern
                        ).all()
                    ):
                        return 4, None
                    result[
                        i * pattern.shape[0] : (i + 1) * pattern.shape[0],
                        j * pattern.shape[1] : (j + 1) * pattern.shape[1],
                    ] = pattern

        return 0, result

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        if initial:
            for _, block in self.sample["train"][k]["blocks"]["arrays"].items():
                pattern = block["array"]
                if (
                    target_image.shape[0] % pattern.shape[0] != 0
                    or target_image.shape[1] % pattern.shape[1] != 0
                ):
                    continue
                for _, mask_path in self.sample["train"][k]["masks"]["arrays"].items():
                    mask = mask_path["array"]
                    if (
                        target_image.shape[0] != pattern.shape[0] * mask.shape[0]
                        or target_image.shape[1] != pattern.shape[1] * mask.shape[1]
                    ):
                        continue
                    for background_color in range(10):
                        if not (target_image == background_color).any():
                            continue
                        params = {"background_color": background_color}

                        status, predict = self.predict_output(
                            original_image, params, pattern=pattern, mask=mask
                        )

                        if status == 0 and (predict == target_image).all():
                            for pattern_params in block["params"]:
                                for mask_params in mask_path["params"]:
                                    for color_dict in self.sample["train"][k]["colors"][
                                        background_color
                                    ]:
                                        params = {
                                            "background_color": color_dict,
                                            "mask": mask_params,
                                            "pattern": pattern_params,
                                        }
                                        local_candidates.append(params)

        else:
            block_cache = self.sample["train"][k]["blocks"]
            mask_cache = self.sample["train"][k]["masks"]
            color_scheme = self.sample["train"][k]

            for candidate in self.solution_candidates:
                status, pattern = get_predict(
                    original_image,
                    candidate["pattern"],
                    block_cache=block_cache,
                    color_scheme=color_scheme,
                )
                if status != 0:
                    continue
                if (
                    target_image.shape[0] % pattern.shape[0] != 0
                    or target_image.shape[1] % pattern.shape[1] != 0
                ):
                    continue

                status, mask = get_mask_from_block_params(
                    original_image,
                    candidate["mask"],
                    block_cache=block_cache,
                    mask_cache=mask_cache,
                    color_scheme=color_scheme,
                )
                if status != 0:
                    continue
                if (
                    target_image.shape[0] != pattern.shape[0] * mask.shape[0]
                    or target_image.shape[1] != pattern.shape[1] * mask.shape[1]
                ):
                    continue
                background_color = get_color(
                    candidate["background_color"], color_scheme["colors"]
                )
                if not (target_image == background_color).any():
                    continue
                params = {"background_color": background_color}

                status, predict = self.predict_output(
                    original_image,
                    params,
                    pattern=pattern,
                    mask=mask,
                    target_image=target_image,
                )

                if status == 0 and (predict == target_image).all():
                    local_candidates.append(candidate)

        return self.update_solution_candidates(local_candidates, initial)


class colors(predictor):
    """returns colors as answers"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)
        # self.type = params["type"]

    def predict_output(self, image, params):
        if params["type"] == "one":
            return 0, np.array([[params["color"]]])
        if params["type"] == "mono_vert":
            num = (image == params["color"]).sum()
            if num <= 0:
                return 7, 0
            return 0, np.array([[params["color"]] * num])
        if params["type"] == "mono_hor":
            num = (image == params["color"]).sum()
            if num <= 0:
                return 7, 0
            return 0, np.array([[params["color"] * num]])
        if params["type"] == "several_linear":
            colors_array = np.rot90(
                np.array(
                    [
                        params["color_scheme"]["colors_sorted"][
                            params["i"] : params["i"] + params["size"]
                        ]
                    ]
                ),
                params["rotate"],
            )
            return 0, colors_array

        if params["type"] == "square":
            colors_array = np.zeros((params["size"] * 2 + 1, params["size"] * 2 + 1))
            if (
                len(params["color_scheme"]["colors_sorted"])
                < params["i"] + params["size"]
            ):
                return 6, None
            if params["direct"] == 0:
                for j in range(params["size"]):
                    colors_array[j:-j] = params["color_scheme"]["colors_sorted"][
                        params["i"] + j
                    ]
            else:
                for j in range(params["size"]):
                    colors_array[j:-j] = params["color_scheme"]["colors_sorted"][::-1][
                        params["i"] + j
                    ]
            return 0, colors_array

        return 9, None

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        if target_image.shape[0] == 1 and target_image.shape[1] == 1:
            params = {"type": "one", "color": int(target_image[0, 0])}
            local_candidates = local_candidates + self.add_candidates_list(
                original_image, target_image, self.sample["train"][k], params
            )
        if target_image.shape[0] == 1:
            params = {"type": "mono_vert", "color": int(target_image[0, 0])}
            local_candidates = local_candidates + self.add_candidates_list(
                original_image, target_image, self.sample["train"][k], params
            )
        if target_image.shape[1] == 1:
            params = {"type": "mono_hor", "color": int(target_image[0, 0])}
            local_candidates = local_candidates + self.add_candidates_list(
                original_image, target_image, self.sample["train"][k], params
            )
        if target_image.shape[0] == 1 or target_image.shape[1] == 1:
            size = target_image.shape[0] * target_image.shape[1]
            if not (size > self.sample["train"][k]["colors_num"]):
                size_diff = self.sample["train"][k]["colors_num"] - size
                for i in range(size_diff + 1):
                    for rotate in range(4):
                        params = {
                            "type": "several_linear",
                            "i": i,
                            "rotate": rotate,
                            "size": size,
                        }
                        local_candidates = local_candidates + self.add_candidates_list(
                            original_image,
                            target_image,
                            self.sample["train"][k],
                            params,
                        )
        if target_image.shape[0] == target_image.shape[1]:
            size = target_image.shape[0] // 2
            if not (size > self.sample["train"][k]["colors_num"]):
                size_diff = self.sample["train"][k]["colors_num"] - size
                for i in range(size_diff + 1):
                    for direct in range(2):
                        params = {
                            "type": "square",
                            "i": i,
                            "direct": direct,
                            "size": size,
                        }
                        local_candidates = local_candidates + self.add_candidates_list(
                            original_image,
                            target_image,
                            self.sample["train"][k],
                            params,
                        )

        return self.update_solution_candidates(local_candidates, initial)


class gravity(predictor):
    """move non_background objects toward something"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        result = np.rot90(image.copy(), params["rotate"])

        steps = params["steps"]
        if steps == "all":
            steps = 10000
        color = params["color"]
        proceed = True
        step = 0
        while proceed and step < steps:
            step += 1
            proceed = False
            for i in range(1, result.shape[0]):
                for j in range(0, result.shape[1]):
                    if params["fill"] == "to_point":
                        if result[-i - 1, j] != color:
                            result[-i, j] = result[-i - 1, j]
                            result[-i - 1, j] = color
                            proceed = True
                    elif result[-i, j] == color and result[-i - 1, j] != color:
                        if params["fill"] == "self":
                            result[-i, j] = result[-i - 1, j]
                        elif params["fill"] == "no":
                            result[-i, j] = result[-i - 1, j]
                            result[-i - 1, j] = color
                        else:
                            result[-i, j] = params["fill_color"]
                        proceed = True

        return 0, np.rot90(result, -params["rotate"])

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        if original_image.shape != target_image.shape:
            return 5, None

        for color in self.sample["train"][k]["colors_sorted"]:
            for rotate in range(0, 4):
                for steps in ["all"] + list(range(max(original_image.shape))):
                    for fill in ["no", "self", "color", "to_point"]:
                        for i, fill_color in enumerate(
                            self.sample["train"][k]["colors_sorted"]
                        ):
                            if fill == "color" and fill_color == color:
                                continue
                            params = {
                                "color": color,
                                "rotate": rotate,
                                "steps": steps,
                                "fill_color": fill_color if fill == "color" else 0,
                                "fill": fill,
                            }

                            local_candidates = (
                                local_candidates
                                + self.add_candidates_list(
                                    original_image,
                                    target_image,
                                    self.sample["train"][k],
                                    params,
                                )
                            )
                            if fill != "color":
                                break
        return self.update_solution_candidates(local_candidates, initial)


class gravity_blocks(predictor):
    """move non_background objects toward something"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def get_block_mask(self, image, i, j, block_type, structure_type):
        if structure_type == 0:
            structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        else:
            structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        if block_type == "same_color":
            color = image[i, j]
            masks, n_masks = ndimage.label(image == color, structure=structure)
        elif block_type == "not_bg":
            color = image[i + 1, j]
            masks, n_masks = ndimage.label(image != color, structure=structure)

        mask = masks == masks[i, j]
        return 0, mask

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        result = np.rot90(image.copy(), params["rotate"])

        color = params["color"]
        proceed = True
        step = 0
        while proceed:
            step += 1
            proceed = False
            for i in range(1, result.shape[0]):
                for j in range(0, result.shape[1]):
                    if result[-i, j] == color and result[-i - 1, j] != color:
                        block_color = result[-i - 1, j]
                        status, mask = self.get_block_mask(
                            result,
                            -i - 1,
                            j,
                            params["block_type"],
                            params["structure_type"],
                        )
                        if status != 0:
                            continue

                        while not (mask[-1] == True).any():
                            moved_mask = np.roll(mask, 1, axis=0)
                            if (
                                result[np.logical_and(moved_mask, moved_mask != mask)]
                                == color
                            ).all():
                                temp = result[mask]
                                result[mask] = color
                                result[moved_mask] = temp
                                proceed = True
                                mask = moved_mask
                            else:
                                break

        return 0, np.rot90(result, -params["rotate"])

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        if original_image.shape != target_image.shape:
            return 5, None

        for color in self.sample["train"][k]["colors_sorted"]:
            for rotate in range(0, 4):
                for block_type in ["same_color", "not_bg"]:
                    for structure_type in [0, 1]:
                        params = {
                            "color": color,
                            "rotate": rotate,
                            "block_type": block_type,
                            "structure_type": structure_type,
                        }

                        local_candidates = local_candidates + self.add_candidates_list(
                            original_image,
                            target_image,
                            self.sample["train"][k],
                            params,
                        )
        return self.update_solution_candidates(local_candidates, initial)


class gravity_blocks_2_color(gravity_blocks):
    """move non_background objects toward something"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def find_gravity_color(self, image, gravity_color):
        mask = image == gravity_color
        if not (mask).any():
            return 1, None, None

        max_hor = mask.max(0)
        max_vert = mask.max(1)

        if max_hor.sum() == 1 and max_vert.sum() > 1:
            color_type = "vert"
            num = np.argmax(max_hor)
        elif max_hor.sum() > 1 and max_vert.sum() == 1:
            color_type = "hor"
            num = np.argmax(max_vert)
        else:
            return 2, None, None

        return 0, color_type, num

    def predict_partial_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        result = np.rot90(image.copy(), params["rotate"])

        color = params["color"]
        proceed = True
        step = 0
        while proceed:
            step += 1
            proceed = False
            for i in range(1, result.shape[0]):
                for j in range(0, result.shape[1]):
                    if result[-i, j] == color and result[-i - 1, j] != color:
                        block_color = result[-i - 1, j]
                        status, mask = self.get_block_mask(
                            result,
                            -i - 1,
                            j,
                            params["block_type"],
                            params["structure_type"],
                        )
                        if status != 0:
                            continue

                        while not (mask[-1] == True).any():
                            moved_mask = np.roll(mask, 1, axis=0)
                            if (
                                result[np.logical_and(moved_mask, moved_mask != mask)]
                                == color
                            ).all():
                                temp = result[mask]
                                result[mask] = color
                                result[moved_mask] = temp
                                proceed = True
                                mask = moved_mask
                            else:
                                break

        return 0, np.rot90(result, -params["rotate"])

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        color = params["color"]
        status, color_type, num = self.find_gravity_color(
            image, params["gravity_color"]
        )
        if status != 0:
            return status, None

        if color_type == "hor":
            top_image = image[:num]
            new_params = params.copy()
            new_params["rotate"] = 0
            status, top_image = self.predict_partial_output(top_image, new_params)
            if status != 0:
                return status, None
            bottom_image = image[num + 1 :]
            new_params = params.copy()
            new_params["rotate"] = 2
            status, bottom_image = self.predict_partial_output(bottom_image, new_params)
            if status != 0:
                return status, None
            result = image.copy()
            result[:num] = top_image
            result[num + 1 :] = bottom_image
            result[
                :, np.logical_not((image == params["gravity_color"]).max(0))
            ] = params["color"]
        elif color_type == "vert":
            left_image = image[:, :num]
            new_params = params.copy()
            new_params["rotate"] = 3
            status, left_image = self.predict_partial_output(left_image, new_params)
            if status != 0:
                return status, None
            right_image = image[:, num + 1 :]
            new_params = params.copy()
            new_params["rotate"] = 1
            status, right_image = self.predict_partial_output(right_image, new_params)
            if status != 0:
                return status, None
            result = image.copy()
            result[:, :num] = left_image
            result[:, num + 1 :] = right_image
            result[np.logical_not((image == params["gravity_color"]).max(1))] = params[
                "color"
            ]

        return 0, result

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        if original_image.shape != target_image.shape:
            return 5, None

        for color in self.sample["train"][k]["colors_sorted"]:
            for gravity_color in self.sample["train"][k]["colors_sorted"]:
                for block_type in ["same_color", "not_bg"]:
                    for structure_type in [0, 1]:
                        params = {
                            "color": color,
                            "gravity_color": gravity_color,
                            "block_type": block_type,
                            "structure_type": structure_type,
                        }

                        local_candidates = local_candidates + self.add_candidates_list(
                            original_image,
                            target_image,
                            self.sample["train"][k],
                            params,
                        )
        return self.update_solution_candidates(local_candidates, initial)


class gravity2color(gravity_blocks_2_color):
    """move non_background objects toward something"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_partial_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        result = np.rot90(image.copy(), params["rotate"])

        steps = params["steps"]
        if steps == "all":
            steps = 10000
        color = params["color"]
        proceed = True
        step = 0
        while proceed and step < steps:
            step += 1
            proceed = False
            for i in range(1, result.shape[0]):
                for j in range(0, result.shape[1]):
                    if params["fill"] == "to_point":
                        if result[-i - 1, j] != color:
                            result[-i, j] = result[-i - 1, j]
                            result[-i - 1, j] = color
                            proceed = True
                    elif result[-i, j] == color and result[-i - 1, j] != color:
                        if params["fill"] == "self":
                            result[-i, j] = result[-i - 1, j]
                        elif params["fill"] == "no":
                            result[-i, j] = result[-i - 1, j]
                            result[-i - 1, j] = color
                        else:
                            result[-i, j] = params["fill_color"]
                        proceed = True

        return 0, np.rot90(result, -params["rotate"])

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        if original_image.shape != target_image.shape:
            return 5, None

        for color in self.sample["train"][k]["colors_sorted"]:
            for gravity_color in self.sample["train"][k]["colors_sorted"]:
                for steps in ["all"] + list(range(max(original_image.shape))):
                    for fill in ["no", "self", "color", "to_point"]:
                        for i, fill_color in enumerate(
                            self.sample["train"][k]["colors_sorted"]
                        ):
                            if fill == "color" and fill_color == color:
                                continue
                            params = {
                                "color": color,
                                "gravity_color": gravity_color,
                                "steps": steps,
                                "fill_color": fill_color if fill == "color" else 0,
                                "fill": fill,
                            }

                            local_candidates = (
                                local_candidates
                                + self.add_candidates_list(
                                    original_image,
                                    target_image,
                                    self.sample["train"][k],
                                    params,
                                )
                            )
                            if fill != "color":
                                break
        return self.update_solution_candidates(local_candidates, initial)


class eliminate_color(predictor):
    """eliminate parts of some color"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        result = image.copy()

        if params["vert"] == True:
            i = 0
            while i < result.shape[0]:
                if (result[i] == params["color"]).all():
                    result = np.concatenate([result[:i], result[i + 1 :]], 0)
                else:
                    i += 1
        if params["hor"] == True:
            i = 0
            while i < result.shape[1]:
                if (result[:, i] == params["color"]).all():
                    result = np.concatenate([result[:, :i], result[:, i + 1 :]], 1)
                else:
                    i += 1

        return 0, result

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        for color in self.sample["train"][k]["colors_sorted"]:
            for hor in [True, False]:
                for vert in [True, False]:
                    params = {"color": color, "hor": hor, "vert": vert}

                    local_candidates = local_candidates + self.add_candidates_list(
                        original_image, target_image, self.sample["train"][k], params
                    )
        return self.update_solution_candidates(local_candidates, initial)


class eliminate_duplicates(predictor):
    """eliminate rows and colomns if they are the same and near each other"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        result = image.copy()

        if params["vert"] == True:
            i = 0
            while i + 1 < result.shape[0]:
                if (result[i] == result[i + 1]).all():
                    result = np.concatenate([result[:i], result[i + 1 :]], 0)
                elif params["elim_bg"] and (result[i] == params["bg_color"]).all():
                    result = np.concatenate([result[:i], result[i + 1 :]], 0)
                elif params["elim_bg"] and (result[i + 1] == params["bg_color"]).all():
                    result = np.concatenate([result[: i + 1], result[i + 2 :]], 0)
                else:
                    i += 1
        if params["hor"] == True:
            i = 0
            while i + 1 < result.shape[1]:
                if (result[:, i] == result[:, i + 1]).all():
                    result = np.concatenate([result[:, :i], result[:, i + 1 :]], 1)
                elif params["elim_bg"] and (result[:, i] == params["bg_color"]).all():
                    result = np.concatenate([result[:, :i], result[:, i + 1 :]], 1)
                elif (
                    params["elim_bg"] and (result[:, i + 1] == params["bg_color"]).all()
                ):
                    result = np.concatenate([result[:, : i + 1], result[:, i + 2 :]], 1)
                else:
                    i += 1

        return 0, result

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        for hor in [True, False]:
            for vert in [True, False]:
                for elim_bg in [True, False]:
                    for bg_color in self.sample["train"][k]["colors_sorted"]:
                        params = {
                            "hor": hor,
                            "vert": vert,
                            "elim_bg": elim_bg,
                            "bg_color": bg_color,
                        }

                        local_candidates = local_candidates + self.add_candidates_list(
                            original_image,
                            target_image,
                            self.sample["train"][k],
                            params,
                        )
                        if not elim_bg:
                            break
        return self.update_solution_candidates(local_candidates, initial)


class connect_dots(predictor):
    """connect dost of same color, on one line"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_part(self, image, params, part_type, result=None):
        if result is None:
            result = image.copy()
        if part_type == "vert":
            if params["vert"] == True:
                for i in range(result.shape[0]):
                    line_mask = image[i] == params["color"]
                    if (line_mask).sum() >= params["min_in_line"]:
                        indices = [x for x in range(len(line_mask)) if line_mask[x]]
                        if params["fill_all"]:
                            result[i, indices[0] + 1 : indices[-1]] = params[
                                "fill_color"
                            ]
                        else:
                            for j in range(len(indices) - 1):
                                result[i, indices[j] + 1 : indices[j + 1]] = params[
                                    "fill_color"
                                ]
        elif part_type == "hor":
            if params["hor"] == True:
                for i in range(result.shape[1]):
                    line_mask = image[:, i] == params["color"]
                    if (line_mask).sum() >= params["min_in_line"]:
                        indices = [x for x in range(len(line_mask)) if line_mask[x]]
                        if params["fill_all"]:
                            result[indices[0] + 1 : indices[-1], i] = params[
                                "fill_color"
                            ]
                        else:
                            for j in range(len(indices) - 1):
                                result[indices[j] + 1 : indices[j + 1], i] = params[
                                    "fill_color"
                                ]

        return result

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        if params["vert_first"]:
            result = self.predict_part(image, params, "vert")
            result = self.predict_part(image, params, "hor", result)
        else:
            result = self.predict_part(image, params, "hor")
            result = self.predict_part(image, params, "vert", result)

        return 0, result

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        for color in self.sample["train"][k]["colors_sorted"]:
            for hor in [True, False]:
                for vert in [True, False]:
                    for fill_color in range(10):
                        for fill_all in [True, False]:
                            for vert_first in [True, False]:
                                for min_in_line in [2, 3, 4]:
                                    params = {
                                        "color": color,
                                        "hor": hor,
                                        "vert": vert,
                                        "fill_color": fill_color,
                                        "fill_all": fill_all,
                                        "vert_first": vert_first,
                                        "min_in_line": min_in_line,
                                    }

                                    local_candidates = (
                                        local_candidates
                                        + self.add_candidates_list(
                                            original_image,
                                            target_image,
                                            self.sample["train"][k],
                                            params,
                                        )
                                    )
        return self.update_solution_candidates(local_candidates, initial)


class connect_dots_all_colors(predictor):
    """connect dost of same color, on one line"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_part(self, image, params, part_type, result=None):
        if result is None:
            result = image.copy()
        if part_type == "vert":
            if params["vert"] == True:
                for color in range(10):
                    if color == params["background_color"]:
                        continue
                    if params["fill_self"]:
                        fill_color = color
                    else:
                        fill_color = params["fill_color"]
                    for i in range(result.shape[0]):
                        line_mask = image[i] == color
                        if (line_mask).sum() >= 2:
                            indices = [x for x in range(len(line_mask)) if line_mask[x]]
                            if params["fill_all"]:
                                result[i, indices[0] + 1 : indices[-1]] = fill_color
                            else:
                                for j in range(len(indices) - 1):
                                    result[
                                        i, indices[j] + 1 : indices[j + 1]
                                    ] = fill_color
        elif part_type == "hor":
            if params["hor"] == True:
                for color in range(10):
                    if color == params["background_color"]:
                        continue
                    if params["fill_self"]:
                        fill_color = color
                    else:
                        fill_color = params["fill_color"]
                    for i in range(result.shape[1]):
                        line_mask = image[:, i] == color
                        if (line_mask).sum() >= 2:
                            indices = [x for x in range(len(line_mask)) if line_mask[x]]
                            if params["fill_all"]:
                                result[indices[0] + 1 : indices[-1], i] = fill_color
                            else:
                                for j in range(len(indices) - 1):
                                    result[
                                        indices[j] + 1 : indices[j + 1], i
                                    ] = fill_color

        return result

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        if params["vert_first"]:
            result = self.predict_part(image, params, "vert")
            result = self.predict_part(image, params, "hor", result)
        else:
            result = self.predict_part(image, params, "hor")
            result = self.predict_part(image, params, "vert", result)

        return 0, result

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        for background_color in self.sample["train"][k]["colors_sorted"]:
            for hor in [True, False]:
                for vert in [True, False]:
                    for fill_self in [True, False]:
                        for fill_all in [False, True]:
                            for vert_first in [True, False]:
                                for fill_color in range(10):
                                    params = {
                                        "background_color": background_color,
                                        "hor": hor,
                                        "vert": vert,
                                        "fill_color": fill_color,
                                        "fill_all": fill_all,
                                        "vert_first": vert_first,
                                        "fill_self": fill_self,
                                    }

                                    local_candidates = (
                                        local_candidates
                                        + self.add_candidates_list(
                                            original_image,
                                            target_image,
                                            self.sample["train"][k],
                                            params,
                                        )
                                    )
                                    if fill_self:
                                        break
        return self.update_solution_candidates(local_candidates, initial)


class reconstruct_mosaic(predictor):
    """reconstruct mosaic"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def check_surface(self, image, i, j, block, color, bg):
        b = (image.shape[0] - i) // block.shape[0] + int(
            ((image.shape[0] - i) % block.shape[0]) > 0
        )
        r = (image.shape[1] - j) // block.shape[1] + int(
            ((image.shape[1] - j) % block.shape[1]) > 0
        )
        t = (i) // block.shape[0] + int((i) % block.shape[0] > 0)
        l = (j) // block.shape[1] + int((j) % block.shape[1] > 0)

        # print(image.shape, t,b, l,r, i, j)

        full_image = (
            np.ones(((b + t) * block.shape[0], (r + l) * block.shape[1])) * color
        )
        start_i = (block.shape[0] - i) % block.shape[0]
        start_j = (block.shape[1] - j) % block.shape[1]

        full_image[
            start_i : start_i + image.shape[0], start_j : start_j + image.shape[1]
        ] = image

        blocks = []
        for k in range((b + t)):
            for n in range((r + l)):
                new_block = full_image[
                    k * block.shape[0] : (k + 1) * block.shape[0],
                    n * block.shape[1] : (n + 1) * block.shape[1],
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

        if (new_block == color).any() and not bg:
            temp_array = np.concatenate([new_block, new_block], 0)
            temp_array = np.concatenate([temp_array, temp_array], 1)
            for k in range(new_block.shape[0]):
                for n in range(new_block.shape[1]):
                    current_array = temp_array[
                        k : k + new_block.shape[0], n : n + new_block.shape[1]
                    ]
                    mask = np.logical_and(new_block != color, current_array != color)
                    if (new_block == current_array)[mask].all():
                        new_block[new_block == color] = current_array[
                            new_block == color
                        ]
        if (new_block == color).any() and not bg:
            return 3, None

        for k in range((b + t)):
            for n in range((r + l)):
                full_image[
                    k * block.shape[0] : (k + 1) * block.shape[0],
                    n * block.shape[1] : (n + 1) * block.shape[1],
                ] = new_block

        result = full_image[
            start_i : start_i + image.shape[0], start_j : start_j + image.shape[1]
        ]
        return 0, result

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        itteration_list1 = list(range(2, sum(image.shape)))
        if params["big_first"]:
            itteration_list1 = list(
                range(
                    2,
                    (image != params["color"]).max(1).sum()
                    + (image != params["color"]).max(0).sum()
                    + 1,
                )
            )
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
                # for i_min in range(min(image.shape[0], i_size)):
                #     for j_min in range(min(image.shape[1], j_size)):
                block = image[0 : 0 + i_size, 0 : 0 + j_size]
                status, predict = self.check_surface(
                    image, 0, 0, block, params["color"], params["have_bg"]
                )
                if status != 0:
                    continue
                return 0, predict

        return 1, None

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)
        if original_image.shape != target_image.shape:
            return 1, None

        if initial:
            directions = ["vert", "hor", "all"]
            big_first_options = [True, False]
            have_bg_options = [True, False]
        else:
            directions = list(
                set([params["direction"] for params in self.solution_candidates])
            )
            big_first_options = list(
                set([params["big_first"] for params in self.solution_candidates])
            )
            have_bg_options = list(
                set([params["have_bg"] for params in self.solution_candidates])
            )

        for color in self.sample["train"][k]["colors_sorted"]:
            for direction in directions:
                for big_first in big_first_options:
                    for have_bg in have_bg_options:
                        if (target_image == color).any() and not have_bg:
                            continue
                        params = {
                            "color": color,
                            "direction": direction,
                            "big_first": big_first,
                            "have_bg": have_bg,
                        }

                        local_candidates = local_candidates + self.add_candidates_list(
                            original_image,
                            target_image,
                            self.sample["train"][k],
                            params,
                        )
        return self.update_solution_candidates(local_candidates, initial)


class reconstruct_mosaic_rr(predictor):
    """reconstruct mosaic"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def check_surface(self, image, i, j, color, direction, reuse_edge, keep_bg):
        blocks = []
        blocks.append(image[i:, j:])
        if direction == "rotate":
            if reuse_edge:
                blocks.append(np.rot90(image[: i + 1, j:], 1))
                blocks.append(np.rot90(image[i:, : j + 1], -1))
                blocks.append(np.rot90(image[: i + 1, : j + 1], 2))
            else:
                blocks.append(np.rot90(image[:i, j:], 1))
                blocks.append(np.rot90(image[i:, :j], -1))
                blocks.append(np.rot90(image[:i, :j], 2))
        elif direction == "reflect":
            if reuse_edge:
                blocks.append(image[: i + 1, j:][::-1, :])
                blocks.append(image[i:, : j + 1][:, ::-1])
                blocks.append(image[: i + 1, : j + 1][::-1, ::-1])
            else:
                blocks.append(image[:i, j:][::-1, :])
                blocks.append(image[i:, :j][:, ::-1])
                blocks.append(image[:i, :j][::-1, ::-1])

        size = [0, 0]
        size[0] = max([x.shape[0] for x in blocks])
        size[1] = max([x.shape[1] for x in blocks])

        full_block = np.ones(size) * color
        for curr_block in blocks:
            temp_block = full_block[: curr_block.shape[0], : curr_block.shape[1]]
            mask = np.logical_and(temp_block != color, curr_block != color)
            if (temp_block == curr_block)[mask].all():
                temp_block[temp_block == color] = curr_block[temp_block == color]
            else:
                return 2, None

        if not keep_bg and (full_block == color).any():
            temp_block = full_block[: min(size), : min(size)]
            temp_block[temp_block == color] = temp_block.T[temp_block == color]

        if not keep_bg and (full_block == color).any():
            return 3, None

        result = image.copy()
        result[i:, j:] = full_block[: blocks[0].shape[0], : blocks[0].shape[1]]
        if direction == "rotate":
            if reuse_edge:
                result[: i + 1, j:] = np.rot90(
                    full_block[: blocks[1].shape[0], : blocks[1].shape[1]], -1
                )
                result[i:, : j + 1] = np.rot90(
                    full_block[: blocks[2].shape[0], : blocks[2].shape[1]], 1
                )
                result[: i + 1, : j + 1] = np.rot90(
                    full_block[: blocks[3].shape[0], : blocks[3].shape[1]], 2
                )
            else:
                result[:i, j:] = np.rot90(
                    full_block[: blocks[1].shape[0], : blocks[1].shape[1]], -1
                )
                result[i:, :j] = np.rot90(
                    full_block[: blocks[2].shape[0], : blocks[2].shape[1]], 1
                )
                result[:i, :j] = np.rot90(
                    full_block[: blocks[3].shape[0], : blocks[3].shape[1]], 2
                )
        elif direction == "reflect":
            if reuse_edge:
                result[: i + 1, j:] = full_block[
                    : blocks[1].shape[0], : blocks[1].shape[1]
                ][::-1, :]
                result[i:, : j + 1] = full_block[
                    : blocks[2].shape[0], : blocks[2].shape[1]
                ][:, ::-1]
                result[: i + 1, : j + 1] = full_block[
                    : blocks[3].shape[0], : blocks[3].shape[1]
                ][::-1, ::-1]
            else:
                result[:i, j:] = full_block[: blocks[1].shape[0], : blocks[1].shape[1]][
                    ::-1, :
                ]
                result[i:, :j] = full_block[: blocks[2].shape[0], : blocks[2].shape[1]][
                    :, ::-1
                ]
                result[:i, :j] = full_block[: blocks[3].shape[0], : blocks[3].shape[1]][
                    ::-1, ::-1
                ]

        return 0, result

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        itteration_list1 = list(range(2, sum(image.shape)))
        for size in itteration_list1:
            itteration_list = list(range(1, size))
            for i in itteration_list:
                j = size - i
                if j < 1 or i < 1:
                    continue
                status, predict = self.check_surface(
                    image,
                    i,
                    j,
                    params["color"],
                    params["direction"],
                    params["reuse_edge"],
                    params["keep_bg"],
                )
                if status != 0:
                    continue
                return 0, predict

        return 1, None

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)
        if original_image.shape != target_image.shape:
            return 1, None

        if initial:
            directions = ["rotate", "reflect"]
            reuse_edge_options = [True, False]
            keep_bg_options = [True, False]
        else:
            directions = list(
                set([params["direction"] for params in self.solution_candidates])
            )
            reuse_edge_options = list(
                set([params["reuse_edge"] for params in self.solution_candidates])
            )
            keep_bg_options = list(
                set([params["keep_bg"] for params in self.solution_candidates])
            )

        for color in self.sample["train"][k]["colors_sorted"]:
            for direction in directions:
                for reuse_edge in reuse_edge_options:
                    for keep_bg in keep_bg_options:
                        if not keep_bg and (target_image == color).any():
                            continue
                        params = {
                            "color": color,
                            "direction": direction,
                            "reuse_edge": reuse_edge,
                            "keep_bg": keep_bg,
                        }

                        local_candidates = local_candidates + self.add_candidates_list(
                            original_image,
                            target_image,
                            self.sample["train"][k],
                            params,
                        )
        return self.update_solution_candidates(local_candidates, initial)


class reconstruct_mosaic_extract(reconstruct_mosaic):
    """reconstruct mosaic"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""

        mask = image == params["color"]
        sum0 = mask.sum(0)
        sum1 = mask.sum(1)
        indices0 = np.arange(len(sum1))[sum1 > 0]
        indices1 = np.arange(len(sum0))[sum0 > 0]

        itteration_list1 = list(range(2, sum(image.shape)))
        if params["big_first"]:
            itteration_list1 = list(
                range(
                    2,
                    (image != params["color"]).max(1).sum()
                    + (image != params["color"]).max(0).sum()
                    + 1,
                )
            )
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
                # for i_min in range(min(image.shape[0], i_size)):
                #     for j_min in range(min(image.shape[1], j_size)):
                block = image[0 : 0 + i_size, 0 : 0 + j_size]
                status, predict = self.check_surface(
                    image, 0, 0, block, params["color"], params["have_bg"]
                )
                if status != 0:
                    continue
                predict = predict[
                    indices0.min() : indices0.max() + 1,
                    indices1.min() : indices1.max() + 1,
                ]
                return 0, predict

        return 1, None

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        if initial:
            directions = ["vert", "hor", "all"]
            big_first_options = [True, False]
            have_bg_options = [True, False]
        else:
            directions = list(
                set([params["direction"] for params in self.solution_candidates])
            )
            big_first_options = list(
                set([params["big_first"] for params in self.solution_candidates])
            )
            have_bg_options = list(
                set([params["have_bg"] for params in self.solution_candidates])
            )

        for color in self.sample["train"][k]["colors_sorted"]:
            mask = original_image == color
            sum0 = mask.sum(0)
            sum1 = mask.sum(1)

            if len(np.unique(sum0)) != 2 or len(np.unique(sum1)) != 2:
                continue
            if target_image.shape[0] != max(sum0) or target_image.shape[1] != max(sum1):
                continue
            for direction in directions:
                for big_first in big_first_options:
                    for have_bg in have_bg_options:
                        if (target_image == color).any() and not have_bg:
                            continue
                        params = {
                            "color": color,
                            "direction": direction,
                            "big_first": big_first,
                            "have_bg": have_bg,
                        }

                        local_candidates = local_candidates + self.add_candidates_list(
                            original_image,
                            target_image,
                            self.sample["train"][k],
                            params,
                        )
        return self.update_solution_candidates(local_candidates, initial)


class reconstruct_mosaic_rr_extract(reconstruct_mosaic_rr):
    """reconstruct mosaic"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""

        mask = image == params["color"]
        sum0 = mask.sum(0)
        sum1 = mask.sum(1)
        indices0 = np.arange(len(sum1))[sum1 > 0]
        indices1 = np.arange(len(sum0))[sum0 > 0]

        itteration_list1 = list(range(2, sum(image.shape)))
        for size in itteration_list1:
            itteration_list = list(range(1, size))
            for i in itteration_list:
                j = size - i
                if j < 1 or i < 1:
                    continue
                status, predict = self.check_surface(
                    image,
                    i,
                    j,
                    params["color"],
                    params["direction"],
                    params["reuse_edge"],
                    params["keep_bg"],
                )
                if status != 0:
                    continue
                predict = predict[
                    indices0.min() : indices0.max() + 1,
                    indices1.min() : indices1.max() + 1,
                ]
                return 0, predict

        return 1, None

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        if initial:
            directions = ["rotate", "reflect"]
            reuse_edge_options = [True, False]
            keep_bg_options = [True, False]
        else:
            directions = list(
                set([params["direction"] for params in self.solution_candidates])
            )
            reuse_edge_options = list(
                set([params["reuse_edge"] for params in self.solution_candidates])
            )
            keep_bg_options = list(
                set([params["keep_bg"] for params in self.solution_candidates])
            )

        for color in self.sample["train"][k]["colors_sorted"]:
            mask = original_image == color
            sum0 = mask.sum(0)
            sum1 = mask.sum(1)
            if len(np.unique(sum0)) != 2 or len(np.unique(sum1)) != 2:
                continue
            if target_image.shape[0] != max(sum0) or target_image.shape[1] != max(sum1):
                continue
            for direction in directions:
                for reuse_edge in reuse_edge_options:
                    for keep_bg in keep_bg_options:
                        if not keep_bg and (target_image == color).any():
                            continue
                        params = {
                            "color": color,
                            "direction": direction,
                            "reuse_edge": reuse_edge,
                            "keep_bg": keep_bg,
                        }

                        local_candidates = local_candidates + self.add_candidates_list(
                            original_image,
                            target_image,
                            self.sample["train"][k],
                            params,
                        )
        return self.update_solution_candidates(local_candidates, initial)


# TODO: test inside_block
class inside_block(reconstruct_mosaic_rr):
    """reconstruct mosaic"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""

        status, block = get_predict(
            image,
            params["block"],
            block_cache=params["block_cache"],
            color_scheme=params["color_scheme"],
        )
        if status != 0:
            return 1, None

        i = params["i"]
        if i >= min(block.shape) / 2:
            return 1, None
        return 0, block[i:-i, i:-i]

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        if initial:
            for k, block in self.sample["train"][k]["blocks"]["arrays"].items():
                array = block["array"]
                diff_0 = -target_image.shape[0] + array.shape[0]
                diff_1 = -target_image.shape[1] + array.shape[1]
                if diff_1 != diff_0 or diff_1 <= 0 or diff_0 % 2 != 0:
                    continue
                if (
                    array[diff_0 // 2 : -diff_0 // 2, diff_0 // 2 : -diff_0 // 2]
                    == target_image
                ).all():
                    # print(block)
                    for params in block["params"]:
                        local_candidates.append({"i": diff_0 // 2, "block": params})
        else:
            for candidate in self.solution_candidates:
                local_candidates = local_candidates + self.add_candidates_list(
                    original_image, target_image, self.sample["train"][k], candidate
                )

        return self.update_solution_candidates(local_candidates, initial)


class fill_lines(predictor):
    """fill the whole horizontal and/or vertical lines with one color"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""

        result = image.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] == params["color"]:
                    if params["vert"]:
                        result[i] = params["fill_color"]
                    if params["hor"]:
                        result[:, j] = params["fill_color"]
        if params["keep"]:
            result[image == params["keep_color"]] = params["keep_color"]
        else:
            result[image != params["keep_color"]] = image[image != params["keep_color"]]
        return 0, result

    def process_one_sample(self, k, initial=False):
        """ processes k train sample and updates self.solution_candidates"""
        local_candidates = []
        original_image, target_image = self.get_images(k)

        if original_image.shape != target_image.shape:
            return 2

        for color in self.sample["train"][k]["colors_sorted"]:
            for hor in [True, False]:
                for vert in [True, False]:
                    if not hor and not vert:
                        continue
                    for fill_color in range(10):
                        for keep in [True, False]:
                            for keep_color in self.sample["train"][k]["colors_sorted"]:
                                params = {
                                    "color": color,
                                    "hor": hor,
                                    "vert": vert,
                                    "fill_color": fill_color,
                                    "keep_color": keep_color,
                                    "keep": keep,
                                }

                                local_candidates = (
                                    local_candidates
                                    + self.add_candidates_list(
                                        original_image,
                                        target_image,
                                        self.sample["train"][k],
                                        params,
                                    )
                                )
        return self.update_solution_candidates(local_candidates, initial)


# TODO: targets based logic
# TODO: pattern transfer
# TODO: mask to answer
