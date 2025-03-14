import os
from multiprocessing import Pool

import dill
import numpy as np
import pandas as pd
import wandb

# def download_run(run_id, project_name):
#     if os.path.exists(f"./results_data/tmp/{run_id}.pkl"):
#         with open(f"./results_data/tmp/{run_id}.pkl", "rb") as f:
#             run_df = dill.load(f)
#         return run_id, run_df
#     else:
#         print("downloading: ", run_id)
#         run = wandb.Api().run(f"lucacorbucci/{project_name}/{run_id}")
#         dataset = run.config["dataset"]
#         epochs = run.config["epochs"]
#         if "synthesizer" in run.config:
#             synthetiser = run.config["synthesizer"]
#         else:
#             synthetiser = None
#         if "samples_to_generate" in run.config:
#             samples_to_generate = run.config["samples_to_generate"]
#         else:
#             samples_to_generate = None

#         run_df = pd.DataFrame(run.scan_history())
#         run_df["dataset"] = dataset
#         run_df["epochs"] = epochs
#         if synthetiser is not None:
#             run_df["synthesizer"] = synthetiser
#         if samples_to_generate is not None:
#             run_df["samples_to_generate"] = samples_to_generate

#         with open(f"./results_data/tmp/{run_id}.pkl", "wb") as f:
#             dill.dump(run_df, f)
#         return run_id, run_df


def download_run(run_id, project_name):
    if os.path.exists(f"./results_data/tmp/{run_id}.pkl"):
        with open(f"./results_data/tmp/{run_id}.pkl", "rb") as f:
            run_df = dill.load(f)
        return run_id, run_df
    else:
        run = wandb.Api().run(f"lucacorbucci/{project_name}/{run_id}")
        # top_k = run.config["top_k"]

        run_df = pd.DataFrame(run.scan_history())
        # run_df["top_k"] = top_k

        return run_id, run_df


def download_runs(project_name):
    if not os.path.exists(f"./results_data/data_{project_name}.pkl"):
        project_details = wandb.Api().runs(f"lucacorbucci/{project_name}")
        run_ids = [run.id for run in project_details]
        print(run_ids)

        with Pool(50) as pool:
            results = pool.starmap(download_run, [(run_id, project_name) for run_id in run_ids])

        project_data = {}
        for run_id, run_df in results:
            if run_df is not None:
                run_name = wandb.Api().run(f"lucacorbucci/{project_name}/{run_id}").name
                if run_name not in project_data:
                    project_data[run_name] = []
                project_data[run_name].append(run_df)

        with open(f"./results_data/data_{project_name}.pkl", "wb") as f:
            dill.dump(project_data, f)
    else:
        with open(f"./results_data/data_{project_name}.pkl", "rb") as f:
            project_data = dill.load(f)
    return project_data


project_data = download_runs(project_name="new_metrics_computation")
project_name = "new_metrics_computation"
