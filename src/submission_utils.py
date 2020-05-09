import json
import os

from src.preprocessing import preprocess_sample
from src.utils import show_sample, matrix2answer
import matplotlib as mpl
from matplotlib import pyplot as plt

from pebble import ProcessPool
from pebble.common import ProcessExpired
from concurrent.futures import TimeoutError

from tqdm.notebook import tqdm
from functools import partial


def process_file(
    file_path,
    PATH,
    predictors,
    preprocess_params=None,
    show_results=True,
    break_after_answer=False,
):
    with open(os.path.join(PATH, file_path), "r") as file:
        sample = json.load(file)

    submission_list = []
    sample = preprocess_sample(sample, params=preprocess_params)
    answers = []

    for predictor in predictors:
        result, answer = predictor(sample)
        if result == 0:
            if show_results:
                show_sample(sample)
            for j in range(len(answer[0])):
                if not matrix2answer(answer[0][j]) in answers:
                    if show_results and j < 3:
                        plt.matshow(
                            answer[0][j],
                            cmap="Set3",
                            norm=mpl.colors.Normalize(vmin=0, vmax=9),
                        )
                        plt.show()
                        print(file_path, matrix2answer(answer[0][j]))
                    answers = set(list(answers) + [matrix2answer(answer[0][j])])
            for j in range(len(answer)):
                for k in range(len(answer[j])):
                    submission_list.append(
                        {
                            "output_id": file_path[:-5] + "_" + str(j),
                            "output": matrix2answer(answer[j][k]),
                        }
                    )

            if break_after_answer:
                break


def pebble_run(
    files_list,
    PATH,
    predictors,
    preprocess_params=None,
    show_results=True,
    break_after_answer=False,
    processes=20,
    timeout=100,
):
    func = partial(
        process_file,
        PATH=PATH,
        predictors=predictors,
        preprocess_params=preprocess_params,
        show_results=show_results,
        break_after_answer=break_after_answer,
    )

    with ProcessPool(processes) as pool:
        with tqdm(total=len(files_list)) as pbar:
            future = pool.map(func, files_list, timeout=timeout)
            iterator = future.result()

            while True:
                try:
                    result = next(iterator)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print("function took longer than %d seconds" % error.args[1])
                except KeyboardInterrupt:
                    print("got Ctrl+C")
                except ProcessExpired:
                    print("Exited with error")
                finally:
                    pbar.update()
