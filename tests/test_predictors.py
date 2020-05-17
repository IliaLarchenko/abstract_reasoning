import os
from src.predictors import *
from src.preprocessing import *


def check(predictor_class, params, file_path, DATA_PATH, preprocessing_params):
    with open(os.path.join(DATA_PATH, file_path), "r") as file:
        sample = json.load(file)

    sample = preprocess_sample(sample, params=preprocessing_params, process_whole_ds=True)
    predictor = predictor_class(params=params)

    result, answer = predictor(sample)
    if result == 0:
        for i in range(len(answer)):
            test_solved = False
            j = 0
            n = 3
            while j < n and j < len(answer[i]):
                if j > 0 and (answer[i][j].shape == answer[i][j - 1].shape) and (answer[i][j] == answer[i][j - 1]).all():
                    n += 1
                    j += 1
                    continue
                if (answer[i][j].shape == np.uint8(sample["test"][i]["output"]).shape) and (
                    answer[i][j] == np.uint8(sample["test"][i]["output"])
                ).all():
                    test_solved = True
                    break
                j += 1
            if not test_solved:
                return False
        return True

    return False


def test_predictor():
    for id, predictor_class, params, file_path, DATA_PATH, preprocessing_params in [
        (1, fill, {}, "4258a5f9.json", "data/training", ["initial"]),
        (2, fill, {}, "bb43febb.json", "data/training", ["initial"]),
        (3, puzzle, {"intersection": 0}, "a416b8f3.json", "data/training", ["initial"]),
        (4, puzzle, {"intersection": 0}, "59341089.json", "data/evaluation", ["initial", "rotate", "transpose"]),
        (5, puzzle, {"intersection": 0}, "25ff71a9.json", "data/training", ["initial", "halves", "cut_edges"]),
        (6, puzzle, {"intersection": 0}, "e9afcf9a.json", "data/training", ["initial", "corners", "cut_edges"]),
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
        (18, puzzle, {"intersection": 0}, "9a4bb226.json", "data/evaluation", ["initial", "block_with_side_colors"]),
        (19, puzzle, {"intersection": 0}, "8efcae92.json", "data/training", ["initial", "block_with_side_colors"]),
        (20, colors, {}, "f8b3ba0a.json", "data/training", ["initial"]),
        (21, pattern_from_blocks, {}, "0692e18c.json", "data/evaluation", ["initial", "initial_masks"]),
        (22, gravity, {}, "1e0a9b12.json", "data/training", ["initial"]),
        (23, gravity, {}, "3906de3d.json", "data/training", ["initial"]),
        (24, gravity, {}, "d037b0a7.json", "data/training", ["initial"]),
        (25, eliminate_color, {}, "68b67ca3.json", "data/evaluation", ["initial"]),
        (26, gravity_blocks, {}, "5ffb2104.json", "data/evaluation", ["initial"]),
        (27, gravity_blocks, {}, "d282b262.json", "data/evaluation", ["initial"]),
        (28, gravity_blocks_2_color, {}, "6ad5bdfd.json", "data/evaluation", ["initial"]),
        (29, gravity2color, {}, "f83cb3f6.json", "data/evaluation", ["initial"]),
        (30, gravity2color, {}, "13713586.json", "data/evaluation", ["initial"]),
        (
            31,
            fill,
            {"pattern": np.array([[False, True, False], [True, False, True], [False, True, False]])},
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
        (42, connect_dots_all_colors, {}, "40853293.json", "data/training", ["initial"]),
        (43, connect_dots_all_colors, {}, "22168020.json", "data/training", ["initial"]),
        (44, connect_dots_all_colors, {}, "070dd51e.json", "data/evaluation", ["initial"]),
        (45, connect_dots_all_colors, {}, "22eb0ac0.json", "data/training", ["initial"]),
        (46, eliminate_duplicates, {}, "90c28cc7.json", "data/training", ["initial"]),
        (47, reconstruct_mosaic, {}, "f823c43c.json", "data/evaluation", ["initial"]),
        (48, reconstruct_mosaic, {}, "d8c310e9.json", "data/training", ["initial"]),
        (49, reconstruct_mosaic, {}, "8eb1be9a.json", "data/training", ["initial"]),
        (50, reconstruct_mosaic, {}, "05269061.json", "data/training", ["initial"]),
        (51, reconstruct_mosaic, {}, "484b58aa.json", "data/training", ["initial"]),
        (52, reconstruct_mosaic, {}, "29ec7d0e.json", "data/training", ["initial"]),
        (53, reconstruct_mosaic, {}, "ca8f78db.json", "data/evaluation", ["initial"]),
        (
            54,
            reconstruct_mosaic,
            {"skip_train": 1, "roll": (0, 1), "rrr_input": False},
            "caa06a1f.json",
            "data/training",
            ["initial"],
        ),
        (55, reconstruct_mosaic_rr, {}, "b8825c91.json", "data/training", ["initial"]),
        (56, reconstruct_mosaic_rr, {}, "903d1b4a.json", "data/evaluation", ["initial"]),
        (57, reconstruct_mosaic_rr, {}, "af22c60d.json", "data/evaluation", ["initial"]),
        (58, reconstruct_mosaic_rr, {}, "3631a71a.json", "data/training", ["initial"]),
        (59, reconstruct_mosaic_rr_extract, {}, "dc0a314f.json", "data/training", ["initial"]),
        (60, reconstruct_mosaic_rr_extract, {}, "9ecd008a.json", "data/training", ["initial"]),
        (61, reconstruct_mosaic_rr_extract, {}, "ff805c23.json", "data/training", ["initial"]),
        (62, reconstruct_mosaic_rr_extract, {}, "67b4a34d.json", "data/evaluation", ["initial"]),
        (63, reconstruct_mosaic_rr_extract, {}, "f4081712.json", "data/evaluation", ["initial"]),
        (64, reconstruct_mosaic_rr_extract, {}, "e66aafb8.json", "data/evaluation", ["initial"]),
        (65, reconstruct_mosaic_rr_extract, {}, "0934a4d8.json", "data/evaluation", ["initial"]),
        (66, reconstruct_mosaic_rr_extract, {}, "de493100.json", "data/evaluation", ["initial"]),
        (67, reconstruct_mosaic_rr_extract, {}, "f9012d9b.json", "data/training", ["initial"]),
        (68, reconstruct_mosaic_rr, {}, "f9d67f8b.json", "data/evaluation", ["initial"]),
        (69, fill_lines, {}, "319f2597.json", "data/evaluation", ["initial"]),
        (70, fill_lines, {}, "4f537728.json", "data/evaluation", ["initial"]),
        (71, replace_column, {}, "3618c87e.json", "data/training", ["initial"]),
        (72, replace_column, {}, "0d3d703e.json", "data/training", ["initial"]),
        (73, replace_column, {"rotate": 1}, "a85d4709.json", "data/training", ["initial"]),
        (74, replace_column, {"rotate": 1}, "f45f5ca7.json", "data/evaluation", ["initial"]),
        (
            75,
            pattern_from_blocks,
            {},
            "8f2ea7aa.json",
            "data/training",
            ["initial", "max_area_covered", "initial_masks"],
        ),
        (76, mask_to_block, {"mask_num": 1}, "f76d97a5.json", "data/training", ["initial", "initial_masks"]),
        (
            77,
            mask_to_block,
            {"mask_num": 2},
            "bda2d7a6.json",
            "data/training",
            ["initial", "background", "initial_masks"],
        ),
        (77, fill, {}, "dc1df850.json", "data/training", ["initial"]),
        (78, fill, {}, "f0df5ff0.json", "data/evaluation", ["initial"]),
        (79, fill, {}, "fc754716.json", "data/evaluation", ["initial", "background"]),
        (80, extend_targets, {}, "7447852a.json", "data/training", ["initial"]),
        (81, extend_targets, {}, "332efdb3.json", "data/evaluation", ["initial"]),
        (
            82,
            puzzle,
            {"intersection": 0},
            "3979b1a8.json",
            "data/evaluation",
            ["initial", "background", "cut_edges", "resize"],
        ),  # strange, but works
        (
            83,
            mask_to_block,
            {"mask_num": 3},
            "ea9794b1.json",
            "data/evaluation",
            ["initial", "corners", "initial_masks"],
        ),
        (84, mask_to_block, {"mask_num": 3}, "3d31c5b3.json", "data/evaluation", ["initial", "k_part", "initial_masks"]),
        # (85, mask_to_block,{"mask_num": 3},
        #   "6a11f6da.json", "data/evaluation", ["initial", "k_part", "initial_masks"]), #very slow
        (
            86,
            mask_to_block,
            {"mask_num": 2},
            "d47aa2ff.json",
            "data/evaluation",
            ["initial", "grid_cells", "additional_masks", "initial_masks"],
        ),
        (87, colors, {}, "85c4e7cd.json", "data/training", ["initial"]),
        (88, fill_lines, {}, "c1d99e64.json", "data/training", ["initial"]),
    ]:
        assert check(predictor_class, params, file_path, DATA_PATH, preprocessing_params) == True, f"Error in {id}"
