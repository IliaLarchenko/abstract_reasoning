import numpy as np
from src.preprocessing import get_color, get_color_scheme
from src.functions import filter_list_of_dicts


class predictor:
    def init(self, params=None, preprocess_params=None):
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


class fill_outer(predictor):
    """fill all pixels around all pixels with particular color with new color"""

    def init(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        result = image.copy()
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
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
                    self.sample["processed_train"][k]["colors"],
                    params,
                )
        return self.update_solution_candidates(local_candidates, initial)


class fill_inner(fill_outer):
    """fill the pixels with color is all neighbour colors have background color"""

    def init(self, params=None, preprocess_params=None):
        super().__init__(params, preprocess_params)

    def predict_output(self, image, params):
        """ predicts 1 output image given input image and prediction params"""
        result = image.copy()
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
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

        return 0, result
