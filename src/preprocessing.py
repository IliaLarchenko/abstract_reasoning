def find_grid(image):
    "Looks for the grid in image and returns color and size"
    grid_color = -1
    size = [0, 0]
    # TODO: border = False

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

    return (grid_color, size)
