import os
from src.predictors import *
from src.preprocessing import *


def check(predictor_class, params, file_path, DATA_PATH):
    with open(os.path.join(DATA_PATH, file_path), "r") as file:
        sample = json.load(file)
    sample = preprocess_sample(sample)

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
    for id, predictor_class, params, file_path, DATA_PATH in [
        (1, fill, {"type": "outer"}, "4258a5f9.json", "data/training"),
        (2, fill, {"type": "inner"}, "bb43febb.json", "data/training"),
        (3, puzzle, {"intersection": 0}, "a416b8f3.json", "data/training"),
    ]:
        assert (
            check(predictor_class, params, file_path, DATA_PATH) == True
        ), f"Error in {id}"
