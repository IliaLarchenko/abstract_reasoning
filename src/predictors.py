import numpy as np
from src.preprocessing import get_color, get_color_scheme


class predictor:
    def init(self, params, preprocess_params):
        self.params = params
        self.preprocess_params = preprocess_params
        self.solution_candidates = []

    def retrive_params_values(self, params, color_scheme):
        new_params = {}
        for k, v in params.items():
            if k[-5:] == "color":
                new_params[k] = get_color(v, color_scheme["colors"])
            else:
                new_params[k] = v
        return new_params

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
            status = self.process_one_sample(self, k, initial=(k == 0))
            if status != 0:
                return 1

        if len(self.solution_candidates) == 0:
            return 2

        return 0

    def __call__(self):
        status = self.process_full_train()
        if status != 0:
            return status, None

        answers = []
        for _ in self.sample["test"]:
            answers.append([])

        result_generated = False
        for test_n, test_data in enumerate(self.sample["test"]):
            original_image = self.get_images(self, test_n, train=False)
            color_scheme = get_color_scheme(original_image)
            for params_dict in self.solution_candidates:
                status, params = self.retrive_params_values(params_dict, color_scheme)
                if status != 0:
                    continue
                status, prediction = self.predict_output(original_image, color_scheme)
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
            return 1, None
        for background_color in range(10):
            if not (target_image == background_color).any():
                continue
            for fill_color in range(10):
                if not (target_image == fill_color).any():
                    continue
                if not (target_image == original_image)[
                    np.logical_and(
                        target_image != background_color, target_image != fill_color
                    )
                ].all():
                    continue
                params = {
                    "background_color": background_color,
                    "fill_color": fill_color,
                }
                prediction = self.predict_output(original_image, params)
                if (prediction == target_image).all():
                    for color_fill_dict in self.sample["processed_train"][k]["colors"][
                        fill_color
                    ]:
                        for color_background_dict in self.sample["processed_train"][k][
                            "colors"
                        ][background_color]:
                            self.solution_candidates.append(
                                {
                                    "fill_color": color_fill_dict,
                                    "background_color": color_background_dict,
                                }
                            )
        return 0
