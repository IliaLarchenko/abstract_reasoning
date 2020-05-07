import numpy as np
from src.preprocessing import get_color, get_color_scheme
from src.functions import filter_list_of_dicts
from src.preprocessing import find_grid, get_predict


class predictor:
    def __init__(self, params=None, preprocess_params=None):
        self.params = params
        self.preprocess_params = preprocess_params
        self.solution_candidates = []

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

    def get_images(self, k, train=True):
        if train:
            original_image = np.uint8(self.sample["train"][k]["input"])
            target_image = np.uint8(self.sample["train"][k]["output"])
            return original_image, target_image
        else:
            original_image = np.uint8(self.sample["test"][k]["input"])
            return original_image

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        return 1, None

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

    def add_candidates_list(self, image, target_image, colors, params):
        status, prediction = self.predict_output(image, params)
        if status != 0 or not (prediction == target_image).all():
            return []

        result = [params.copy()]
        for k, v in params.copy().items():
            if k[-5:] == "color":
                temp_result = result.copy()
                result = []
                for dict in temp_result:
                    for color_dict in colors[v]:
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
        status = self.process_full_train()
        if status != 0:
            return status, None

        answers = []
        for _ in self.sample["test"]:
            answers.append([])

        result_generated = False
        for test_n, test_data in enumerate(self.sample["test"]):
            original_image = self.get_images(test_n, train=False)
            color_scheme = get_color_scheme(original_image)
            for params_dict in self.solution_candidates:
                status, params = self.retrive_params_values(params_dict, color_scheme)
                if status != 0:
                    continue
                status, prediction = self.predict_output(original_image, params)
                if status != 0:
                    continue

                answers[test_n].append(prediction)
                result_generated = True

        if result_generated:
            return 0, answers
        else:
            return 3, None


class fill(predictor):
    """inner fills all pixels around all pixels with particular color with new color
    outer fills the pixels with fill color if all neighbour colors have background color"""

    def __init__(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)
        self.type = params["type"]  # inner or outer

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        result = image.copy()
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                if self.type == "outer":
                    if image[i, j] == params["fill_color"]:
                        result[i - 1 : i + 2, j - 1 : j + 2][
                            np.array(
                                [
                                    [True, True, True],
                                    [True, False, True],
                                    [True, True, True],
                                ]
                            )
                        ] = params["background_color"]
                elif self.type == "inner":
                    if (
                        image[i - 1 : i + 2, j - 1 : j + 2][
                            np.array(
                                [
                                    [True, True, True],
                                    [True, False, True],
                                    [True, True, True],
                                ]
                            )
                        ]
                        == params["background_color"]
                    ).all():
                        result[i, j] = params["fill_color"]
                else:
                    return 6, None

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
                params = {
                    "background_color": background_color,
                    "fill_color": fill_color,
                }

                local_candidates = local_candidates + self.add_candidates_list(
                    original_image,
                    target_image,
                    self.sample["train"][k]["colors"],
                    params,
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
            for i in range(1, t_n):
                for j in range(1, t_m):
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
                n, m = array.shape

                if i == 0 and j == 0:
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
        local_candidates = []
        original_image, target_image = self.get_images(k)

        candidates_num = 0
        t_n, t_m = target_image.shape
        color_scheme = get_color_scheme(original_image)
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
                                    start_n : start_n + n, start_m : j * start_m + m
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
            color_scheme = get_color_scheme(original_image)
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
                        answers[test_n].append(prediction)
                        result_generated = True

        if result_generated:
            return 0, answers
        else:
            return 3, None


# TODO: fixed_pattern + self_pattern
# TODO: rotate_roll_reflect to class
# TODO: fill pattern - more general surface type
# TODO: reconstruct pattern
# TODO: reconstruct pattern
# TODO: colors functions
