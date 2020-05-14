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
        (1, fill, {}, "4258a5f9.json", "data/training", ["initial"]),
        (2, fill, {}, "bb43febb.json", "data/training", ["initial"]),
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
        (15, fill, {}, "42a50994.json", "data/training", ["initial"]),
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
        (22, gravity, {}, "1e0a9b12.json", "data/training", ["initial"]),
        (23, gravity, {}, "3906de3d.json", "data/training", ["initial"]),
        (24, gravity, {}, "d037b0a7.json", "data/training", ["initial"]),
        (25, eliminate_color, {}, "68b67ca3.json", "data/evaluation", ["initial"]),
        (26, gravity_blocks, {}, "5ffb2104.json", "data/evaluation", ["initial"]),
        (27, gravity_blocks, {}, "d282b262.json", "data/evaluation", ["initial"]),
        (
            28,
            gravity_blocks_2_color,
            {},
            "6ad5bdfd.json",
            "data/evaluation",
            ["initial"],
        ),
        (29, gravity2color, {}, "f83cb3f6.json", "data/evaluation", ["initial"]),
        (30, gravity2color, {}, "13713586.json", "data/evaluation", ["initial"]),
        (
            31,
            fill,
            {
                "pattern": np.array(
                    [[False, True, False], [True, False, True], [False, True, False]]
                )
            },
            "aedd82e4.json",
            "data/training",
            ["initial"],
        ),
        (32, eliminate_duplicates, {}, "eb5a1d5d.json", "data/training", ["initial"]),
        (33, eliminate_duplicates, {}, "746b3537.json", "data/training", ["initial"]),
        (34, eliminate_duplicates, {}, "e1baa8a4.json", "data/evaluation", ["initial"]),
        (35, eliminate_duplicates, {}, "ce8d95cc.json", "data/evaluation", ["initial"]),
        (36, connect_dots, {}, "dbc1a6ce.json", "data/training", ["initial"]),
        (37, connect_dots, {}, "253bf280.json", "data/training", ["initial"]),
        (38, connect_dots, {}, "ba97ae07.json", "data/training", ["initial"]),
        (39, connect_dots, {}, "a699fb00.json", "data/training", ["initial"]),
        (40, connect_dots, {}, "ded97339.json", "data/training", ["initial"]),
        (41, connect_dots, {}, "aa18de87.json", "data/evaluation", ["initial"]),
    ]:
        assert (
            check(predictor_class, params, file_path, DATA_PATH, preprocessing_params)
            == True
        ), f"Error in {id}"
