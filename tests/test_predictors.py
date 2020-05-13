import os
from src.predictors import *
from src.preprocessing import *


def check(predictor_class, params, file_path, DATA_PATH, preprocessing_params):
    with open(os.path.join(DATA_PATH, file_path), "r") as file:
        sample = json.load(file)

    sample = preprocess_sample(sample, params=preprocessing_params)
    predictor = predictor_class(params=params)

    result, answer = predictor(sample)
    if result == 0:
        for i in range(len(answer)):
            test_solved = False
            for j in range(min(len(answer[i]), 3)):
                result = (answer[i][j] == np.uint8(sample["test"][i]["output"])).all()
                if result:
                    test_solved = True
                    break
            if not test_solved:
                return False
        return True

    return False


def test_predictor():
    for id, predictor_class, params, file_path, DATA_PATH, preprocessing_params in [
        (1, fill, {"type": "outer"}, "4258a5f9.json", "data/training", ["initial"]),
        (2, fill, {"type": "inner"}, "bb43febb.json", "data/training", ["initial"]),
        (3, puzzle, {"intersection": 0}, "a416b8f3.json", "data/training", ["initial"]),
        (
            4,
            puzzle,
            {"intersection": 0},
            "59341089.json",
            "data/evaluation",
            ["initial", "rotate", "transpose"],
        ),
        (
            5,
            puzzle,
            {"intersection": 0},
            "25ff71a9.json",
            "data/training",
            ["initial", "halves", "cut_edges"],
        ),
        (
            6,
            puzzle,
            {"intersection": 0},
            "e9afcf9a.json",
            "data/training",
            ["initial", "corners", "cut_edges"],
        ),
        (
            7,
            puzzle,
            {"intersection": 0},
            "66e6c45b.json",
            "data/evaluation",
            ["initial", "min_max_blocks", "rotate", "cut_edges", "resize"],
        ),
        (8, pattern, None, "ad7e01d0.json", "data/evaluation", ["initial"]),
        (9, pattern, None, "5b6cbef5.json", "data/evaluation", ["initial"]),
        (
            10,
            mask_to_block,
            None,
            "195ba7dc.json",
            "data/evaluation",
            ["initial", "grid_cells", "initial_masks", "additional_masks"],
        ),
        (
            11,
            mask_to_block,
            {"mask_num": 2},
            "cf98881b.json",
            "data/training",
            ["initial", "grid_cells", "initial_masks"],
        ),
        (
            12,
            mask_to_block,
            {"mask_num": 2},
            "ce039d91.json",
            "data/evaluation",
            ["initial", "rotate", "transpose", "initial_masks"],
        ),
        (
            13,
            mask_to_block,
            {"mask_num": 3},
            "a68b268e.json",
            "data/training",
            ["initial", "grid_cells", "initial_masks"],
        ),
        (
            14,
            pattern_from_blocks,
            {},
            "b4a43f3b.json",
            "data/evaluation",
            ["initial", "grid_cells", "resize", "initial_masks"],
        ),
        (15, fill, {"type": "isolated"}, "42a50994.json", "data/training", ["initial"]),
        (16, colors, {}, "d631b094.json", "data/training", ["initial"]),
        (17, colors, {}, "1a2e2828.json", "data/evaluation", ["initial"]),
        (
            18,
            puzzle,
            {"intersection": 0},
            "9a4bb226.json",
            "data/evaluation",
            ["initial", "block_with_side_colors"],
        ),
        (
            19,
            puzzle,
            {"intersection": 0},
            "8efcae92.json",
            "data/training",
            ["initial", "block_with_side_colors"],
        ),
        (20, colors, {}, "f8b3ba0a.json", "data/training", ["initial"]),
        (
            21,
            pattern_from_blocks,
            {},
            "0692e18c.json",
            "data/evaluation",
            ["initial", "initial_masks"],
        ),
        (21, gravity, {}, "1e0a9b12.json", "data/training", ["initial"]),
        (22, gravity, {}, "3906de3d.json", "data/training", ["initial"]),
    ]:
        assert (
            check(predictor_class, params, file_path, DATA_PATH, preprocessing_params)
            == True
        ), f"Error in {id}"
