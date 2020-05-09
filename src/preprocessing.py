import numpy as np
from scipy.stats import mode
from scipy import ndimage
import json
import time
import random


def find_grid(image):
    # Looks for the grid in image and returns color and size
    grid_color = -1
    size = [1, 1]
    # TODO: border = False
    # TODO: bold_grid

    for color in range(10):
        for i in range(size[0] + 1, image.shape[0] // 2 + 1):
            if (image.shape[0] + 1) % i == 0:
                step = (image.shape[0] + 1) // i
                if (image[(step - 1) :: step] == color).all():
                    size[0] = i
                    grid_color = color
        for i in range(size[1] + 1, image.shape[1] // 2 + 1):
            if (image.shape[1] + 1) % i == 0:
                step = (image.shape[1] + 1) // i
                if (image[:, (step - 1) :: step] == color).all():
                    size[1] = i
                    grid_color = color

    return grid_color, size


def find_color_boundaries(array, color):
    # Looks for the boundaries of any color and returns them
    if (array == color).any() == False:
        return None
    ind_0 = np.arange(array.shape[0])
    ind_1 = np.arange(array.shape[1])

    temp_0 = ind_0[(array == color).max(axis=1)]  # axis 0
    min_0, max_0 = temp_0.min(), temp_0.max()

    temp_1 = ind_1[(array == color).max(axis=0)]  # axis 1
    min_1, max_1 = temp_1.min(), temp_1.max()

    return min_0, max_0, min_1, max_1


def get_color_max(image, color):
    # Returns the part of the image inside the color boundaries
    boundaries = find_color_boundaries(image, color)
    if boundaries:
        return (
            0,
            image[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1],
        )
    else:
        return 1, None


def get_grid(image, grid_size, cell):
    """ returns the particular cell form the image with grid"""
    if cell[0] >= grid_size[0] or cell[1] >= grid_size[1]:
        return 1, None
    steps = ((image.shape[0] + 1) // grid_size[0], (image.shape[1] + 1) // grid_size[1])
    block = image[
        steps[0] * cell[0] : steps[0] * (cell[0] + 1) - 1,
        steps[1] * cell[1] : steps[1] * (cell[1] + 1) - 1,
    ]
    return 0, block


def get_half(image, side):
    """ returns the half of the image"""
    if side not in "lrtb":
        return 1, None
    if side == "l":
        return 0, image[:, : (image.shape[1]) // 2]
    if side == "r":
        return 0, image[:, -((image.shape[1]) // 2) :]
    if side == "b":
        return 0, image[-((image.shape[0]) // 2) :, :]
    if side == "t":
        return 0, image[: (image.shape[0]) // 2, :]


def get_corner(image, side):
    """ returns the half of the image"""
    if side not in ["tl", "tr", "bl", "br"]:
        return 1, None
    size = (image.shape[0]) // 2, (image.shape[1]) // 2
    if side == "tl":
        return 0, image[size[0] :, -size[1] :]
    if side == "tr":
        return 0, image[size[0] :, : size[1]]
    if side == "bl":
        return 0, image[: -size[0], : size[1]]
    if side == "br":
        return 0, image[: -size[0], -size[1] :]


def get_rotation(image, k):
    return 0, np.rot90(image, k)


def get_transpose(image):
    return 0, np.transpose(image)


def get_roll(image, shift, axis):
    return 0, np.roll(image, shift=shift, axis=axis)


def get_cut_edge(image, l, r, t, b):
    """deletes pixels from some sided of an image"""
    return 0, image[t : image.shape[0] - b, l : image.shape[1] - r]


def get_resize(image, scale):
    """ resizes image according to scale"""
    if isinstance(scale, int):
        if image.shape[0] % scale != 0 or image.shape[1] % scale != 0:
            return 1, None
        if image.shape[0] < scale or image.shape[1] < scale:
            return 2, None

        arrays = []
        size = image.shape[0] // scale, image.shape[1] // scale
        for i in range(scale):
            for j in range(scale):
                arrays.append(image[i::scale, j::scale])

        result = mode(np.stack(arrays), axis=0).mode[0]
    else:
        size = int(image.shape[0] / scale), int(image.shape[1] / scale)
        result = []
        for i in range(size[0]):
            result.append([])
            for j in range(size[1]):
                result[-1].append(image[int(i * scale), int(j * scale)])

        result = np.uint8(result)

    return 0, result


def get_resize_to(image, size_x, size_y):
    """ resizes image according to scale"""
    scale_x = image.shape[0] // size_x
    scale_y = image.shape[1] // size_y
    if scale_x == 0 or scale_y == 0:
        return 3, None
    if image.shape[0] % scale_x != 0 or image.shape[1] % scale_y != 0:
        return 1, None
    if image.shape[0] < scale_x or image.shape[1] < scale_y:
        return 2, None

    arrays = []
    for i in range(scale_x):
        for j in range(scale_y):
            arrays.append(image[i::scale_x, j::scale_y])

    result = mode(np.stack(arrays), axis=0).mode[0]

    return 0, result


def get_reflect(image, side):
    """ returns images generated by reflections of the input"""
    if side not in ["r", "l", "t", "b", "rt", "rb", "lt", "lb"]:
        return 1, None
    try:
        if side == "r":
            result = np.zeros((image.shape[0], image.shape[1] * 2 - 1))
            result[:, : image.shape[1]] = image
            result[:, -image.shape[1] :] = image[:, ::-1]
        elif side == "l":
            result = np.zeros((image.shape[0], image.shape[1] * 2 - 1))
            result[:, : image.shape[1]] = image[:, ::-1]
            result[:, -image.shape[1] :] = image
        elif side == "b":
            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1]))
            result[: image.shape[0], :] = image
            result[-image.shape[0] :, :] = image[::-1]
        elif side == "t":
            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1]))
            result[: image.shape[0], :] = image[::-1]
            result[-image.shape[0] :, :] = image

        elif side == "rb":
            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1] * 2 - 1))
            result[: image.shape[0], : image.shape[1]] = image
            result[: image.shape[0], -image.shape[1] :] = image[:, ::-1]
            result[-image.shape[0] :, : image.shape[1]] = image[::-1, :]
            result[-image.shape[0] :, -image.shape[1] :] = image[::-1, ::-1]

        elif side == "rt":
            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1] * 2 - 1))
            result[: image.shape[0], : image.shape[1]] = image[::-1, :]
            result[: image.shape[0], -image.shape[1] :] = image[::-1, ::-1]
            result[-image.shape[0] :, : image.shape[1]] = image
            result[-image.shape[0] :, -image.shape[1] :] = image[:, ::-1]

        elif side == "lt":
            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1] * 2 - 1))
            result[: image.shape[0], : image.shape[1]] = image[::-1, ::-1]
            result[: image.shape[0], -image.shape[1] :] = image[::-1, :]
            result[-image.shape[0] :, : image.shape[1]] = image[:, ::-1]
            result[-image.shape[0] :, -image.shape[1] :] = image

        elif side == "lb":
            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1] * 2 - 1))
            result[: image.shape[0], : image.shape[1]] = image[:, ::-1]
            result[: image.shape[0], -image.shape[1] :] = image
            result[-image.shape[0] :, : image.shape[1]] = image[::-1, ::-1]
            result[-image.shape[0] :, -image.shape[1] :] = image[::-1, :]
    except:
        return 2, None

    return 0, result


def get_color_swap(image, color_1, color_2):
    """swapping two colors"""
    if not (image == color_1).any() or not (image == color_2).any():
        return 1, None
    result = image.copy()
    result[image == color_1] = color_2
    result[image == color_2] = color_1
    return 0, result


def get_cut(image, x1, y1, x2, y2):
    if x1 >= x2 or y1 >= y2:
        return 1, None
    else:
        return 0, image[x1:x2, y1:y2]


def get_min_block(image, full=True):
    if full:
        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    else:
        structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    masks, n_masks = ndimage.label(image, structure=structure)
    sizes = [(masks == i).sum() for i in range(1, n_masks + 1)]

    if n_masks == 0:
        return 2, None

    min_n = np.argmin(sizes) + 1

    boundaries = find_color_boundaries(masks, min_n)
    if boundaries:
        return (
            0,
            image[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1],
        )
    else:
        return 1, None


def get_max_block(image, full=True):
    if full:
        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    else:
        structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    masks, n_masks = ndimage.label(image, structure=structure)
    sizes = [(masks == i).sum() for i in range(1, n_masks + 1)]

    if n_masks == 0:
        return 2, None

    max_n = np.argmax(sizes) + 1

    boundaries = find_color_boundaries(masks, max_n)
    if boundaries:
        return (
            0,
            image[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1],
        )
    else:
        return 1, None


def get_color(color_dict, colors):
    """ retrive the absolute number corresponding a color set by color_dict"""
    for i, color in enumerate(colors):
        for data in color:
            equal = True
            for k, v in data.items():
                if k not in color_dict or v != color_dict[k]:
                    equal = False
                    break
            if equal:
                return i
    return -1


def get_mask_from_block(image, color):
    if color in np.unique(image, return_counts=False):
        return 0, image == color
    else:
        return 1, None


def get_mask_from_max_color_coverage(image, color):
    if color in np.unique(image, return_counts=False):
        boundaries = find_color_boundaries(image, color)
        result = (image.copy() * 0).astype(bool)
        result[
            boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1
        ] = True
        return 0, image == color
    else:
        return 1, None


def add_unique_colors(image, result, colors=None):
    """adds information about colors unique for some parts of the image"""
    if colors is None:
        colors = np.unique(image)

    unique_side = [False for i in range(10)]
    unique_corner = [False for i in range(10)]

    half_size = (((image.shape[0] + 1) // 2), ((image.shape[1] + 1) // 2))
    for (image_part, side, unique_list) in [
        (image[: half_size[0]], "bottom", unique_side),
        (image[-half_size[0] :], "top", unique_side),
        (image[:, : half_size[1]], "right", unique_side),
        (image[:, -half_size[1] :], "left", unique_side),
        (image[-half_size[0] :, -half_size[1] :], "tl", unique_corner),
        (image[-half_size[0] :, : half_size[1]], "tr", unique_corner),
        (image[: half_size[0], : half_size[1]], "br", unique_corner),
        (image[: half_size[0], -half_size[1] :], "left", unique_corner),
    ]:
        unique = np.uint8(np.unique(image_part))
        if len(unique) == len(colors) - 1:
            color = [x for x in colors if x not in unique][0]
            unique_list[color] = True
            result["colors"][color].append({"type": "unique", "side": side})

    for i in range(10):
        if unique_corner[i]:
            result["colors"][i].append({"type": "unique", "side": "corner"})
        if unique_side[i]:
            result["colors"][i].append({"type": "unique", "side": "side"})
        if unique_side[i] or unique_corner[i]:
            result["colors"][i].append({"type": "unique", "side": "any"})

    return


def get_color_scheme(image, target_image=None, params=None):
    """processes original image and returns dict color scheme"""
    result = {
        "grid_color": -1,
        "colors": [[], [], [], [], [], [], [], [], [], []],
        "colors_sorted": [],
        "grid_size": [1, 1],
    }

    if params is None:
        params = ["coverage", "unique", "corners", "top", "grid"]

    # preparing colors info

    unique, counts = np.unique(image, return_counts=True)
    colors = [unique[i] for i in np.argsort(counts)]

    result["colors_sorted"] = colors
    result["colors_num"] = len(colors)

    if target_image is None:
        for color in range(10):
            # use abs color value - same for any image
            result["colors"][color].append({"type": "abs", "k": color})
    else:
        unique_target = np.unique(target_image)
        for color in [int(x) for x in set(list(unique_target) + list(unique))]:
            # use abs color value - same for any image
            result["colors"][color].append({"type": "abs", "k": color})

    if "coverage" in params:
        for k, color in enumerate(colors):
            # use k-th colour (sorted by presence on image)
            result["colors"][color].append({"type": "min", "k": k})
            # use k-th colour (sorted by presence on image)
            result["colors"][color].append({"type": "max", "k": len(colors) - k - 1})

    if "unique" in params:
        add_unique_colors(image, result, colors=None)

    if "corners" in params:
        # colors in the corners of images
        result["colors"][image[0, 0]].append({"type": "corner", "side": "tl"})
        result["colors"][image[0, -1]].append({"type": "corner", "side": "tr"})
        result["colors"][image[-1, 0]].append({"type": "corner", "side": "bl"})
        result["colors"][image[-1, -1]].append({"type": "corner", "side": "br"})

    if "top" in params:
        # colors that are on top of other and have full vertical on horizontal line
        for k in range(10):
            mask = image == k
            is_on_top0 = mask.min(axis=0).any()
            is_on_top1 = mask.min(axis=1).any()
            if is_on_top0:
                result["colors"][k].append({"type": "on_top", "side": "0"})
            if is_on_top1:
                result["colors"][k].append({"type": "on_top", "side": "1"})
            if is_on_top1 or is_on_top0:
                result["colors"][k].append({"type": "on_top", "side": "any"})

    if "grid" in params:
        grid_color, grid_size = find_grid(image)
        if grid_color >= 0:
            result["grid_color"] = grid_color
            result["grid_size"] = grid_size
            result["colors"][grid_color].append({"type": "grid"})

    return result


def add_block(target_dict, image, params_list):
    array_hash = hash(image.tostring())
    if array_hash not in target_dict["arrays"]:
        target_dict["arrays"][array_hash] = {"array": image, "params": []}

    for params in params_list:
        params_hash = get_dict_hash(params)
        target_dict["arrays"][array_hash]["params"].append(params)
        target_dict["params"][params_hash] = array_hash


def get_original(image):
    return 0, image


def process_image(
    image,
    max_time=30,
    max_blocks=10000,
    max_masks=1000,
    target_image=None,
    params=None,
    color_params=None,
):
    """processes the original image and returns dict with structured image blocks"""

    all_params = [
        "initial",
        "min_max_blocks",
        "max_area_covered",
        "grid_cells",
        "halves",
        "corners" "rotate",
        "transpose",
        "cut_edges",
        "resize",
        "reflect",
        "cut_parts",
        "swap_colors",
        "initial_masks",
        "additional_masks",
        "coverage_masks",
    ]

    if not params:
        params = all_params

    if "initial" in params:
        # print("initial")
        result = get_color_scheme(image, target_image=target_image, params=color_params)
        result["blocks"] = {"arrays": {}, "params": {}}

        # generating blocks

        # starting with the original image
        add_block(result["blocks"], image, [[{"type": "original"}]])

    start_time = time.time()
    # adding min and max blocks
    if (
        ("min_max_blocks" in params)
        and (time.time() - start_time < max_time)
        and (len(result["blocks"]["arrays"]) < max_blocks)
    ):
        # print("min_max_blocks")
        for full in [True, False]:
            status, block = get_max_block(image, full)
            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                add_block(
                    result["blocks"], block, [[{"type": "max_block", "full": full}]]
                )

    # adding the max area covered by each color
    if ("max_area_covered" in params) and (time.time() - start_time < max_time):
        # print("max_area_covered")
        for color in result["colors_sorted"]:
            status, block = get_color_max(image, color)
            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                params_list = []
                for color_dict in result["colors"][color].copy():
                    params_list.append([{"type": "color_max", "color": color_dict}])
                add_block(result["blocks"], block, params_list)

    # adding grid cells
    if (
        ("grid_cells" in params)
        and (time.time() - start_time < max_time)
        and (len(result["blocks"]["arrays"]) < max_blocks)
    ):
        # print("grid_cells")
        if result["grid_color"] > 0:
            for i in range(result["grid_size"][0]):
                for j in range(result["grid_size"][1]):
                    status, block = get_grid(image, result["grid_size"], (i, j))
                    if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                        add_block(
                            result["blocks"],
                            block,
                            [
                                [
                                    {
                                        "type": "grid",
                                        "grid_size": result["grid_size"],
                                        "cell": [i, j],
                                    }
                                ]
                            ],
                        )

    # adding halves of the images
    if (
        ("halves" in params)
        and (time.time() - start_time < max_time)
        and (len(result["blocks"]["arrays"]) < max_blocks)
    ):
        # print("halves")
        for side in "lrtb":
            status, block = get_half(image, side=side)
            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                add_block(result["blocks"], block, [[{"type": "half", "side": side}]])

    # adding halves of the images
    if (
        ("corners" in params)
        and (time.time() - start_time < max_time)
        and (len(result["blocks"]["arrays"]) < max_blocks)
    ):
        # print("halves")
        for side in ["tl", "tr", "bl", "br"]:
            status, block = get_corner(image, side=side)
            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                add_block(result["blocks"], block, [[{"type": "corner", "side": side}]])

    # TODO: main_blocks!!!
    main_blocks_num = len(result["blocks"])

    # rotate all blocks
    if (
        ("rotate" in params)
        and (time.time() - start_time < max_time)
        and (len(result["blocks"]["arrays"]) < max_blocks)
    ):
        # print("rotate")
        current_blocks = result["blocks"]["arrays"].copy()
        for k in range(1, 4):
            for key, data in current_blocks.items():
                status, block = get_rotation(data["array"], k=k)
                if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                    params_list = [
                        i + [{"type": "rotation", "k": k}] for i in data["params"]
                    ]
                    add_block(result["blocks"], block, params_list)

    # transpose all blocks
    if (
        ("transpose" in params)
        and (time.time() - start_time < max_time)
        and (len(result["blocks"]["arrays"]) < max_blocks)
    ):
        # print("transpose")
        current_blocks = result["blocks"]["arrays"].copy()
        for key, data in current_blocks.items():
            status, block = get_transpose(data["array"])
            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                params_list = [i + [{"type": "transpose"}] for i in data["params"]]
                add_block(result["blocks"], block, params_list)

    # cut edges for all blocks
    if (
        ("cut_edges" in params)
        and (time.time() - start_time < max_time)
        and (len(result["blocks"]["arrays"]) < max_blocks)
    ):
        # print("cut_edges")
        current_blocks = result["blocks"]["arrays"].copy()
        for l, r, t, b in [
            (1, 1, 1, 1),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 1, 0, 0),
            (1, 0, 0, 1),
            (0, 0, 1, 1),
            (0, 1, 1, 0),
        ]:
            if time.time() - start_time < max_time:
                for key, data in current_blocks.items():
                    status, block = get_cut_edge(data["array"], l=l, r=r, t=t, b=b)
                    if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                        params_list = [
                            i + [{"type": "cut_edge", "l": l, "r": r, "t": t, "b": b}]
                            for i in data["params"]
                        ]
                        add_block(result["blocks"], block, params_list)

    # resize all blocks
    if (
        ("resize" in params)
        and (time.time() - start_time < max_time)
        and (len(result["blocks"]["arrays"]) < max_blocks)
    ):
        # print("resize")
        current_blocks = result["blocks"]["arrays"].copy()
        for scale in [2, 3, 1 / 2, 1 / 3]:
            for key, data in current_blocks.items():
                status, block = get_resize(data["array"], scale)
                if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                    params_list = [
                        i + [{"type": "resize", "scale": scale}] for i in data["params"]
                    ]
                    add_block(result["blocks"], block, params_list)

        for size_x, size_y in [(2, 2), (3, 3)]:
            for key, data in current_blocks.items():
                status, block = get_resize_to(data["array"], size_x, size_y)
                if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                    params_list = [
                        i + [{"type": "resize_to", "size_x": size_x, "size_y": size_y}]
                        for i in data["params"]
                    ]
                    add_block(result["blocks"], block, params_list)

    # reflect all blocks
    if (
        ("reflect" in params)
        and (time.time() - start_time < max_time)
        and (len(result["blocks"]["arrays"]) < max_blocks)
    ):
        # print("reflect")
        current_blocks = result["blocks"]["arrays"].copy()
        for side in ["r", "l", "t", "b", "rt", "rb", "lt", "lb"]:
            if time.time() - start_time < max_time:
                for key, data in current_blocks.items():
                    status, block = get_reflect(data["array"], side)
                    if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                        params_list = [
                            i + [{"type": "reflect", "side": side}]
                            for i in data["params"]
                        ]
                        add_block(result["blocks"], block, params_list)

    # cut some parts of images
    if (
        ("cut_parts" in params)
        and (time.time() - start_time < max_time)
        and (len(result["blocks"]["arrays"]) < max_blocks)
    ):
        # print("cut_parts")
        max_x = image.shape[0]
        max_y = image.shape[1]
        min_block_size = 2
        for x1 in range(0, max_x - min_block_size):
            if time.time() - start_time < max_time:
                if max_x - x1 <= min_block_size:
                    continue
                for x2 in range(x1 + min_block_size, max_x):
                    for y1 in range(0, max_y - min_block_size):
                        if max_y - y1 <= min_block_size:
                            continue
                        for y2 in range(y1 + min_block_size, max_y):
                            status, block = get_cut(image, x1, y1, x2, y2)
                            if status == 0:
                                add_block(
                                    result["blocks"],
                                    block,
                                    [
                                        [
                                            {
                                                "type": "cut",
                                                "x1": x1,
                                                "x2": x2,
                                                "y1": y1,
                                                "y2": y2,
                                            }
                                        ]
                                    ],
                                )

    result["masks"] = {"arrays": {}, "params": {}}

    # making one mask for each generated block
    current_blocks = result["blocks"]["arrays"].copy()
    if ("initial_masks" in params) and (time.time() - start_time < max_time * 2):
        # print("initial_masks")
        for key, data in current_blocks.items():
            for color in result["colors_sorted"]:
                status, mask = get_mask_from_block(data["array"], color)
                if status == 0 and mask.shape[0] > 0 and mask.shape[1] > 0:
                    params_list = [
                        {
                            "operation": "none",
                            "params": {
                                "block": i,
                                "color": color_dict,
                                "color_id": int(color),
                            },
                        }
                        for i in data["params"]
                        for color_dict in result["colors"][color]
                    ]
                    add_block(result["masks"], mask, params_list)

        initial_masks = result["masks"]["arrays"].copy()
        for key, mask in initial_masks.items():
            add_block(
                result["masks"],
                np.logical_not(mask["array"]),
                [
                    {"operation": "not", "params": param["params"]}
                    for param in mask["params"]
                ],
            )

    processed = []
    if ("additional_masks" in params) and (time.time() - start_time < max_time * 2):
        # print("additional_masks")
        for key1, mask1 in initial_masks.items():
            processed.append(key1)
            if time.time() - start_time < max_time * 2 and (
                (target_image.shape == mask1["array"].shape)
                or (target_image.shape == mask1["array"].T.shape)
            ):

                # if max_masks * 3 < len(result["masks"]):
                #     break
                for key2, mask2 in initial_masks.items():
                    if key2 in processed:
                        continue
                    if (mask1["array"].shape[0] == mask2["array"].shape[0]) and (
                        mask1["array"].shape[1] == mask2["array"].shape[1]
                    ):
                        params_list_and = []
                        params_list_or = []
                        params_list_xor = []
                        for param1 in mask1["params"]:
                            for param2 in mask2["params"]:
                                params_list_and.append(
                                    {
                                        "operation": "and",
                                        "params": {"mask1": param1, "mask2": param2},
                                    }
                                )
                                params_list_or.append(
                                    {
                                        "operation": "or",
                                        "params": {"mask1": param1, "mask2": param2},
                                    }
                                )
                                params_list_xor.append(
                                    {
                                        "operation": "xor",
                                        "params": {"mask1": param1, "mask2": param2},
                                    }
                                )

                        add_block(
                            result["masks"],
                            np.logical_and(mask1["array"], mask2["array"]),
                            params_list_and,
                        )
                        add_block(
                            result["masks"],
                            np.logical_or(mask1["array"], mask2["array"]),
                            params_list_or,
                        )
                        add_block(
                            result["masks"],
                            np.logical_xor(mask1["array"], mask2["array"]),
                            params_list_xor,
                        )

    # coverage_masks
    if ("coverage_masks" in params) and (time.time() - start_time < max_time * 2):
        # print("coverage_masks")
        for color in result["colors_sorted"][1:]:
            status, mask = get_mask_from_max_color_coverage(image, color)
            if status == 0 and mask.shape[0] > 0 and mask.shape[1] > 0:
                for color_dict in result["colors"][color].copy():
                    params_list = [
                        {"operation": "coverage", "params": {"color": color_dict}}
                        for i in data["params"]
                    ]
                add_block(result["masks"], mask, params_list)

    # swap some colors
    if (
        ("swap_colors" in params)
        and (time.time() - start_time < max_time)
        and (len(result["blocks"]["arrays"]) < max_blocks)
    ):
        # print("swap_colors")
        current_blocks = result["blocks"]["arrays"].copy()
        for i, color_1 in enumerate(result["colors_sorted"][:-1]):
            if time.time() - start_time < max_time:
                for color_2 in result["colors_sorted"][i:]:
                    for key, data in current_blocks.items():
                        status, block = get_color_swap(data["array"], color_1, color_2)
                        if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:
                            for color_dict_1 in result["colors"][color_1].copy():
                                for color_dict_2 in result["colors"][color_2].copy():
                                    params_list = [
                                        i
                                        + [
                                            {
                                                "type": "color_swap",
                                                "color_1": color_dict_1,
                                                "color_2": color_dict_2,
                                            }
                                        ]
                                        for i in data["params"]
                                    ]

                            add_block(result["blocks"], block, params_list)

    if time.time() - start_time > max_time:
        print("Time is over")
    if len(result["blocks"]["arrays"]) >= max_blocks:
        print("Max number of blocks exceeded")
    return result


def get_mask_from_block_params(
    image, params, block_cache=None, mask_cache=None, color_scheme=None
):

    if mask_cache is None:
        mask_cache = {{"arrays": {}, "params": {}}}
    dict_hash = get_dict_hash(params)
    if dict_hash in mask_cache:
        mask = mask_cache["arrays"][mask_cache["params"][dict_hash]]["array"]
        if len(mask) == 0:
            return 1, None
        else:
            return 0, mask

    if params["operation"] == "none":
        status, block = get_predict(
            image, params["params"]["block"], block_cache, color_scheme
        )
        if status != 0:
            add_block(mask_cache, np.array([[]]), [params])
            return 1, None
        if not color_scheme:
            color_scheme = get_color_scheme(image)
        color_num = get_color(params["params"]["color"], color_scheme["colors"])
        if color_num < 0:
            add_block(mask_cache, np.array([[]]), [params])
            return 2, None
        status, mask = get_mask_from_block(block, color_num)
        if status != 0:
            add_block(mask_cache, np.array([[]]), [params])
            return 6, None
        add_block(mask_cache, mask, [params])
        return 0, mask
    elif params["operation"] == "not":
        new_params = params.copy()
        new_params["operation"] = "none"
        status, mask = get_mask_from_block_params(
            image,
            new_params,
            block_cache=block_cache,
            color_scheme=color_scheme,
            mask_cache=mask_cache,
        )
        if status != 0:
            add_block(mask_cache, np.array([[]]), [params])
            return 3, None
        mask = np.logical_not(mask)
        add_block(mask_cache, mask, [params])
        return 0, mask
    elif params["operation"] in ["and", "or", "xor"]:
        new_params = params["params"]["mask1"]
        status, mask1 = get_mask_from_block_params(
            image,
            new_params,
            block_cache=block_cache,
            color_scheme=color_scheme,
            mask_cache=mask_cache,
        )
        if status != 0:
            add_block(mask_cache, np.array([[]]), [params])
            return 4, None
        new_params = params["params"]["mask2"]
        status, mask2 = get_mask_from_block_params(
            image,
            new_params,
            block_cache=block_cache,
            color_scheme=color_scheme,
            mask_cache=mask_cache,
        )
        if status != 0:
            add_block(mask_cache, np.array([[]]), [params])
            return 5, None
        if mask1.shape[0] != mask2.shape[0] or mask1.shape[1] != mask2.shape[1]:
            add_block(mask_cache, np.array([[]]), [params])
            return 6, None
        if params["operation"] == "and":
            mask = np.logical_and(mask1, mask2)
        elif params["operation"] == "or":
            mask = np.logical_or(mask1, mask2)
        elif params["operation"] == "xor":
            mask = np.logical_xor(mask1, mask2)
        add_block(mask_cache, mask, [params])
        return 0, mask
    elif params["operation"] == "coverage":
        if not color_scheme:
            color_scheme = get_color_scheme(image)
        color_num = get_color(params["params"]["color"], color_scheme["colors"])
        if color_num < 0:
            add_block(mask_cache, np.array([[]]), [params])
            return 2, None
        status, mask = get_mask_from_max_color_coverage(image, color_num)
        if status != 0:
            add_block(mask_cache, np.array([[]]), [params])
            return 6, None
        add_block(mask_cache, mask, [params])
        return 0, mask


def get_dict_hash(d):
    return hash(json.dumps(d, sort_keys=True))


def get_predict(image, transforms, block_cache=None, color_scheme=None):
    """ applies the list of transforms to the image"""
    params_hash = get_dict_hash(transforms)
    if params_hash in block_cache["params"]:
        if block_cache["params"][params_hash] is None:
            return 1, None
        else:
            return 0, block_cache["arrays"][block_cache["params"][params_hash]]["array"]

    if not color_scheme:
        color_scheme = get_color_scheme(image)

    if len(transforms) > 1:
        status, previous_image = get_predict(
            image, transforms[:-1], block_cache=block_cache, color_scheme=color_scheme
        )
        if status != 0:
            return status, None
    else:
        previous_image = image

    transform = transforms[-1]
    function = globals()["get_" + transform["type"]]
    params = transform.copy()
    params.pop("type")
    for color_name in ["color", "color_1", "color_2"]:
        if color_name in params:
            params[color_name] = get_color(params[color_name], color_scheme["colors"])
    status, result = function(previous_image, **params)

    if status != 0 or len(result) == 0 or len(result[0]) == 0:
        block_cache["params"][params_hash] = None
        return 1, None

    add_block(block_cache, result, transforms)
    return 0, result


def preprocess_sample(sample, params=None, color_params=None):
    """ make the whole preprocessing for particular sample"""

    original_image = np.uint8(sample["train"][0]["input"])
    target_image = np.uint8(sample["train"][0]["output"])

    sample["train"][0].update(
        process_image(
            original_image,
            target_image=target_image,
            params=params,
            color_params=color_params,
        )
    )

    for n, image in enumerate(sample["train"][1:]):
        original_image = np.uint8(image["input"])
        sample["train"][n + 1].update(
            process_image(original_image, params=["initial"]), color_params=color_params
        )

    for n, image in enumerate(sample["test"]):
        original_image = np.uint8(image["input"])
        sample["test"][n].update(
            process_image(original_image, params=["initial"]), color_params=color_params
        )

    return sample
