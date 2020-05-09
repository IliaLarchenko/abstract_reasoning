import json
import multiprocessing
import os
import time

from src.preprocessing import preprocess_sample
from src.utils import show_sample, matrix2answer
import matplotlib as mpl
from matplotlib import pyplot as plt

from tqdm.notebook import tqdm
from functools import partial


def process_file(
    file_path,
    PATH,
    predictors,
    preprocess_params=None,
    show_results=True,
    break_after_answer=False,
    queue=None,
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
    if queue is not None:
        queue.put(submission_list)


def run_parallel(
    files_list,
    PATH,
    predictors,
    preprocess_params=None,
    show_results=True,
    break_after_answer=False,
    processes=20,
    timeout=10,
):
    process_list = []
    timing_list = []

    queue = multiprocessing.Queue()
    func = partial(
        process_file,
        PATH=PATH,
        predictors=predictors,
        preprocess_params=preprocess_params,
        show_results=show_results,
        break_after_answer=break_after_answer,
        queue=queue,
    )

    with tqdm(total=len(files_list)) as pbar:
        try:
            while True:
                num_finished = 0
                for process, start_time in zip(process_list, timing_list):
                    if process.is_alive():
                        if time.time() - start_time > timeout:
                            process.terminate()
                            process.join(0.1)
                            print("Time out. The process is killed.")
                            num_finished += 1

                    else:
                        num_finished += 1

                if num_finished == len(files_list):
                    pbar.reset()
                    pbar.update(num_finished)
                    time.sleep(0.1)
                    break
                elif len(process_list) - num_finished < processes and len(
                    process_list
                ) < len(files_list):
                    p = multiprocessing.Process(
                        target=func, args=(files_list[len(process_list)],)
                    )
                    process_list.append(p)
                    timing_list.append(time.time())
                    p.start()
                pbar.reset()
                pbar.update(num_finished)
        except KeyboardInterrupt:
            for process in process_list:
                process.terminate()
                process.join(0.1)
            print("Got Ctrl+C")
        except Exception as error:
            for process in process_list:
                process.terminate()
                process.join(0.1)
            print(f"Function raised {error}")
    result = []
    while not queue.empty():
        result = result + queue.get()
    return result
