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
        (1, Fill, {}, "4258a5f9.json", "data/training", ["initial"]),
        (2, Fill, {}, "bb43febb.json", "data/training", ["initial"]),
        (3, Puzzle, {"intersection": 0}, "a416b8f3.json", "data/training", ["initial"]),
        (4, Puzzle, {"intersection": 0}, "59341089.json", "data/evaluation", ["initial", "rotate", "transpose"]),
        (5, Puzzle, {"intersection": 0}, "25ff71a9.json", "data/training", ["initial", "halves", "cut_edges"]),
        (6, Puzzle, {"intersection": 0}, "e9afcf9a.json", "data/training", ["initial", "corners", "cut_edges"]),
        (
            7,
            Puzzle,
            {"intersection": 0},
            "66e6c45b.json",
            "data/evaluation",
            ["initial", "min_max_blocks", "rotate", "cut_edges", "resize"],
        ),
        (8, Pattern, None, "ad7e01d0.json", "data/evaluation", ["initial"]),
        (9, Pattern, None, "5b6cbef5.json", "data/evaluation", ["initial"]),
        (
            10,
            MaskToBlock,
            None,
            "195ba7dc.json",
            "data/evaluation",
            ["initial", "grid_cells", "initial_masks", "additional_masks"],
        ),
        (11, MaskToBlock, {"mask_num": 2}, "cf98881b.json", "data/training", ["initial", "grid_cells", "initial_masks"]),
        (
            12,
            MaskToBlock,
            {"mask_num": 2},
            "ce039d91.json",
            "data/evaluation",
            ["initial", "rotate", "transpose", "initial_masks"],
        ),
        (13, MaskToBlock, {"mask_num": 3}, "a68b268e.json", "data/training", ["initial", "grid_cells", "initial_masks"]),
        (
            14,
            PatternFromBlocks,
            {},
            "b4a43f3b.json",
            "data/evaluation",
            ["initial", "grid_cells", "resize", "initial_masks"],
        ),
        (15, Fill, {}, "42a50994.json", "data/training", ["initial"]),
        (16, Colors, {}, "d631b094.json", "data/training", ["initial"]),
        (17, Colors, {}, "1a2e2828.json", "data/evaluation", ["initial"]),
        (18, Puzzle, {"intersection": 0}, "9a4bb226.json", "data/evaluation", ["initial", "block_with_side_colors"]),
        (19, Puzzle, {"intersection": 0}, "8efcae92.json", "data/training", ["initial", "block_with_side_colors"]),
        (20, Colors, {}, "f8b3ba0a.json", "data/training", ["initial"]),
        (21, PatternFromBlocks, {}, "0692e18c.json", "data/evaluation", ["initial", "initial_masks"]),
        (22, Gravity, {}, "1e0a9b12.json", "data/training", ["initial"]),
        (23, Gravity, {}, "3906de3d.json", "data/training", ["initial"]),
        (24, Gravity, {}, "d037b0a7.json", "data/training", ["initial"]),
        (25, EliminateColor, {}, "68b67ca3.json", "data/evaluation", ["initial"]),
        (26, GravityBlocks, {}, "5ffb2104.json", "data/evaluation", ["initial"]),
        (27, GravityBlocks, {}, "d282b262.json", "data/evaluation", ["initial"]),
        (28, GravityBlocksToColors, {}, "6ad5bdfd.json", "data/evaluation", ["initial"]),
        (29, GravityToColor, {}, "f83cb3f6.json", "data/evaluation", ["initial"]),
        (30, GravityToColor, {}, "13713586.json", "data/evaluation", ["initial"]),
        (
            31,
            Fill,
            {"pattern": np.array([[False, True, False], [True, False, True], [False, True, False]])},
            "aedd82e4.json",
            "data/training",
            ["initial"],
        ),
        (32, EliminateDuplicates, {}, "eb5a1d5d.json", "data/training", ["initial"]),
        (33, EliminateDuplicates, {}, "746b3537.json", "data/training", ["initial"]),
        (34, EliminateDuplicates, {}, "e1baa8a4.json", "data/evaluation", ["initial"]),
        (35, EliminateDuplicates, {}, "ce8d95cc.json", "data/evaluation", ["initial"]),
        (36, ConnectDots, {}, "dbc1a6ce.json", "data/training", ["initial"]),
        (37, ConnectDots, {}, "253bf280.json", "data/training", ["initial"]),
        (38, ConnectDots, {}, "ba97ae07.json", "data/training", ["initial"]),
        (39, ConnectDots, {}, "a699fb00.json", "data/training", ["initial"]),
        (40, ConnectDots, {}, "ded97339.json", "data/training", ["initial"]),
        (41, ConnectDots, {}, "aa18de87.json", "data/evaluation", ["initial"]),
        (42, ConnectDotsAllColors, {}, "40853293.json", "data/training", ["initial"]),
        (43, ConnectDotsAllColors, {}, "22168020.json", "data/training", ["initial"]),
        (44, ConnectDotsAllColors, {}, "070dd51e.json", "data/evaluation", ["initial"]),
        (45, ConnectDotsAllColors, {}, "22eb0ac0.json", "data/training", ["initial"]),
        (46, EliminateDuplicates, {}, "90c28cc7.json", "data/training", ["initial"]),
        (47, ReconstructMosaic, {}, "f823c43c.json", "data/evaluation", ["initial"]),
        (48, ReconstructMosaic, {}, "d8c310e9.json", "data/training", ["initial"]),
        (49, ReconstructMosaic, {}, "8eb1be9a.json", "data/training", ["initial"]),
        (50, ReconstructMosaic, {}, "05269061.json", "data/training", ["initial"]),
        (51, ReconstructMosaic, {}, "484b58aa.json", "data/training", ["initial"]),
        (52, ReconstructMosaic, {}, "29ec7d0e.json", "data/training", ["initial"]),
        (53, ReconstructMosaic, {}, "ca8f78db.json", "data/evaluation", ["initial"]),
        (
            54,
            ReconstructMosaic,
            {"skip_train": 1, "roll": (0, 1), "rrr_input": False},
            "caa06a1f.json",
            "data/training",
            ["initial"],
        ),
        (55, ReconstructMosaicRR, {}, "b8825c91.json", "data/training", ["initial"]),
        (56, ReconstructMosaicRR, {}, "903d1b4a.json", "data/evaluation", ["initial"]),
        (57, ReconstructMosaicRR, {}, "af22c60d.json", "data/evaluation", ["initial"]),
        (58, ReconstructMosaicRR, {}, "3631a71a.json", "data/training", ["initial"]),
        (59, ReconstructMosaicRRExtract, {}, "dc0a314f.json", "data/training", ["initial"]),
        (60, ReconstructMosaicRRExtract, {}, "9ecd008a.json", "data/training", ["initial"]),
        (61, ReconstructMosaicRRExtract, {}, "ff805c23.json", "data/training", ["initial"]),
        (62, ReconstructMosaicRRExtract, {}, "67b4a34d.json", "data/evaluation", ["initial"]),
        (63, ReconstructMosaicRRExtract, {}, "f4081712.json", "data/evaluation", ["initial"]),
        (64, ReconstructMosaicRRExtract, {}, "e66aafb8.json", "data/evaluation", ["initial"]),
        (65, ReconstructMosaicRRExtract, {}, "0934a4d8.json", "data/evaluation", ["initial"]),
        (66, ReconstructMosaicRRExtract, {}, "de493100.json", "data/evaluation", ["initial"]),
        (67, ReconstructMosaicRRExtract, {}, "f9012d9b.json", "data/training", ["initial"]),
        (68, ReconstructMosaicRR, {}, "f9d67f8b.json", "data/evaluation", ["initial"]),
        (69, FillLines, {}, "319f2597.json", "data/evaluation", ["initial"]),
        (70, FillLines, {}, "4f537728.json", "data/evaluation", ["initial"]),
        (71, ReplaceColumn, {}, "3618c87e.json", "data/training", ["initial"]),
        (72, ReplaceColumn, {}, "0d3d703e.json", "data/training", ["initial"]),
        (73, ReplaceColumn, {"rotate": 1}, "a85d4709.json", "data/training", ["initial"]),
        (74, ReplaceColumn, {"rotate": 1}, "f45f5ca7.json", "data/evaluation", ["initial"]),
        (75, PatternFromBlocks, {}, "8f2ea7aa.json", "data/training", ["initial", "max_area_covered", "initial_masks"]),
        (76, MaskToBlock, {"mask_num": 1}, "f76d97a5.json", "data/training", ["initial", "initial_masks"]),
        (77, MaskToBlock, {"mask_num": 2}, "bda2d7a6.json", "data/training", ["initial", "background", "initial_masks"]),
        (77, Fill, {}, "dc1df850.json", "data/training", ["initial"]),
        (78, Fill, {}, "f0df5ff0.json", "data/evaluation", ["initial"]),
        (79, Fill, {}, "fc754716.json", "data/evaluation", ["initial", "background"]),
        (80, ExtendTargets, {}, "7447852a.json", "data/training", ["initial"]),
        (81, ExtendTargets, {}, "332efdb3.json", "data/evaluation", ["initial"]),
        (
            82,
            Puzzle,
            {"intersection": 0},
            "3979b1a8.json",
            "data/evaluation",
            ["initial", "background", "cut_edges", "resize"],
        ),  # strange, but works
        (83, MaskToBlock, {"mask_num": 3}, "ea9794b1.json", "data/evaluation", ["initial", "corners", "initial_masks"]),
        (84, MaskToBlock, {"mask_num": 3}, "3d31c5b3.json", "data/evaluation", ["initial", "k_part", "initial_masks"]),
        (
            85,
            MaskToBlock,
            {"mask_num": 3},
            "6a11f6da.json",
            "data/evaluation",
            ["initial", "k_part", "initial_masks"],
        ),  # very slow
        (
            86,
            MaskToBlock,
            {"mask_num": 2},
            "d47aa2ff.json",
            "data/evaluation",
            ["initial", "grid_cells", "additional_masks", "initial_masks"],
        ),
        (87, Colors, {}, "85c4e7cd.json", "data/training", ["initial"]),
        (88, FillLines, {}, "c1d99e64.json", "data/training", ["initial"]),
        (89, Fill, {}, "6f8cd79b.json", "data/training", ["initial", "background"]),
        (
            90,
            Fill,
            {"pattern": np.array([[False, True, False], [True, False, True], [False, True, False]])},
            "7f4411dc.json",
            "data/training",
            ["initial"],
        ),
        (
            91,
            Fill,
            {"pattern": np.array([[False, True, False], [True, False, True], [False, True, False]])},
            "aedd82e4.json",
            "data/training",
            ["initial"],
        ),
        (
            92,
            Fill,
            {"pattern": np.array([[False, True, False], [True, False, True], [False, True, False]])},
            "67385a82.json",
            "data/training",
            ["initial"],
        ),
        (
            93,
            Fill,
            {"pattern": np.array([[False, True, False], [True, False, True], [False, True, False]])},
            "7f4411dc.json",
            "data/training",
            ["initial"],
        ),
        (
            94,
            Fill,
            {"pattern": np.array([[False, True, False], [True, False, True], [False, True, False]])},
            "6f8cd79b.json",
            "data/training",
            ["initial"],
        ),
        (
            95,
            Fill,
            {"pattern": np.array([[False, True, False], [True, False, True], [False, True, False]])},
            "e0fb7511.json",
            "data/evaluation",
            ["initial"],
        ),
        (
            96,
            Fill,
            {"pattern": np.array([[False, True, False], [True, True, True], [False, True, False]])},
            "7e02026e.json",
            "data/evaluation",
            ["initial"],
        ),
        (97, ReconstructMosaic, {"simple_mode": False}, "92e50de0.json", "data/evaluation", ["initial"]),
        (98, Fill, {"simple_mode": False}, "f0df5ff0.json", "data/evaluation", ["initial"]),
        (99, FillPatternFound, {}, "50f325b5.json", "data/evaluation", ["initial"]),
        (100, FillPatternFound, {}, "bb52a14b.json", "data/evaluation", ["initial"]),
        (101, FillPatternFound, {}, "32597951.json", "data/training", ["initial"]),
        (102, FillPatternFound, {}, "6d75e8bb.json", "data/training", ["initial"]),
        (
            103,
            PutBlockIntoHole,
            {},
            "9f27f097.json",
            "data/evaluation",
            ["initial", "min_max_blocks", "max_area_covered", "rotate", "transpose"],
        ),
        (104, FillPatternFound, {}, "890034e9.json", "data/training", ["initial"]),
        (105, FillPatternFound, {}, "7df24a62.json", "data/training", ["initial"]),
        (106, FillPatternFound, {}, "79369cc6.json", "data/evaluation", ["initial"]),
        (
            107,
            PutBlockIntoHole,
            {},
            "e76a88a6.json",
            "data/training",
            ["initial", "min_max_blocks", "max_area_covered", "block_with_side_colors"],
        ),
        (
            108,
            PutBlockIntoHole,
            {},
            "321b1fc6.json",
            "data/training",
            ["initial", "min_max_blocks", "max_area_covered", "block_with_side_colors"],
        ),
        (
            109,
            PutBlockOnPixel,
            {},
            "88a10436.json",
            "data/training",
            ["initial", "block_with_side_colors", "min_max_blocks", "max_area_covered"],
        ),
        (110, RotateAndCopyBlock, {}, "2697da3f.json", "data/evaluation", ["initial", "max_area_covered"]),
        (111, ReconstructMosaic, {"elim_background": True}, "9c1e755f.json", "data/evaluation", ["initial"]),
        (112, ReconstructMosaic, {"elim_background": True}, "a57f2f04.json", "data/evaluation", ["initial"]),
        (
            113,
            Puzzle,
            {"intersection": 0, "mode": True},
            "5bd6f4ac.json",
            "data/training",
            ["initial", "pixels", "pixel_fixed"],
        ),
        (
            114,
            Puzzle,
            {"intersection": 0, "mode": True},
            "73251a56.json",
            "data/training",
            ["initial", "pixels", "pixel_fixed"],
        ),
        (
            115,
            Puzzle,
            {"intersection": 0, "mode": True},
            "af24b4cc.json",
            "data/evaluation",
            ["initial", "pixels", "pixel_fixed"],
        ),
        (
            116,
            Puzzle,
            {"intersection": 0, "mode": True},
            "ca8de6ea.json",
            "data/evaluation",
            ["initial", "pixels", "pixel_fixed"],
        ),
        (
            117,
            Puzzle,
            {"intersection": 0, "mode": True},
            "4aab4007.json",
            "data/evaluation",
            ["initial", "pixels", "pixel_fixed"],
        ),
        (
            118,
            PuzzlePixel,
            {"intersection": 0, "mode": True},
            "1e97544e.json",
            "data/evaluation",
            ["initial", "pixels", "pixel_fixed"],
        ),
    ]:
        assert check(predictor_class, params, file_path, DATA_PATH, preprocessing_params) == True, f"Error in {id}"
