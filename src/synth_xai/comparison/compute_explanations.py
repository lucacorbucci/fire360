import argparse
import random
import signal
import sys
import warnings
from pathlib import Path
from types import FrameType

import dill
import multiprocess
import numpy as np
import torch
from loguru import logger
from multiprocess import Pool
from sklearn.metrics import accuracy_score

from synth_xai.bb_architectures import MultiClassModel, SimpleModel
from synth_xai.comparison.explainer import Explainer
from synth_xai.comparison.utils_comparison import prepare_data
from synth_xai.explanations.explanation_utils import (
    load_bb,
    make_predictions,
)
from synth_xai.utils import (
    aix_model,
)

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--dataset_name", type=str, default=None, required=True)
parser.add_argument("--bb_path", type=str, default=None, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--explainer_name", type=str, default=None, required=True)
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--validation_seed", type=bool, default=None)
parser.add_argument("--num_processes", type=int, default=20, required=True)
parser.add_argument("--store_path", type=str, default=None, required=True)


def signal_handler(_sig: int, frame: FrameType | None) -> None:
    """
    Function used to handle the SIGINT signal
    """
    logger.info("Gracefully stopping your experiment! Keep calm!")
    # global wandb_run
    # if wandb_run:
    #     wandb_run.finish()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    args = parser.parse_args()

    current_script_path = Path(__file__).resolve().parent

    bb = load_bb(args.bb_path)
    model = aix_model(model=bb)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)
    (
        x_train,
        y_train,
        x_test,
        y_test,
        _,
        _,
        train_df,
        test_data,
        feature_names,
        categorical_feature_names,
        class_names,
    ) = prepare_data(args=args, current_path=current_script_path)

    categorical_features = [
        train_df.columns.get_loc(col) for col in categorical_feature_names if col in train_df.columns
    ]

    explainer = Explainer(
        args=args,
        train_df=train_df,
        feature_names=feature_names,
        categorical_feature_names=categorical_feature_names,
        class_names=class_names,
    )
    multiprocess.set_start_method("spawn")
    num_samples = min(20000, len(test_data))

    def explain_sample(
        explainer: Explainer, model: aix_model, bb: SimpleModel, sample: np.ndarray, y_sample: int, num_features: int
    ) -> tuple[list, int, int]:
        def predict_fn(x: np.ndarray) -> np.ndarray:
            model.model.to("cuda")
            return np.array(model.predict_proba(x))

        explanation, local_pred = explainer.explain_instance(sample, predict_fn, num_features=num_features)
        x_sample = torch.tensor([sample])
        y_sample = torch.tensor([y_sample], dtype=torch.float32)
        bb.to("cuda")
        sample_pred_bb = make_predictions(x_sample, y_sample, bb)
        return explanation, sample_pred_bb[0].item(), local_pred

    args_list = [
        (explainer, model, bb, x_test[sample_idx], y_test[sample_idx], len(feature_names))
        for sample_idx in range(num_samples)
    ]
    with Pool(args.num_processes) as pool:
        results = pool.starmap(explain_sample, args_list)

    explanations, predictions_bb, local_predictions = map(list, zip(*results, strict=False))

    computed_explanations = {}
    for index in range(num_samples):
        computed_explanations[index] = (explanations[index], predictions_bb[index])

    file_name = args.explainer_name + ".pkl"
    store_path = Path(args.store_path) / file_name
    with Path(store_path).open("wb") as f:
        # store the computed_explanations dictionary as a json file
        dill.dump(computed_explanations, f)

    fidelity = accuracy_score(predictions_bb, local_predictions)

    logger.info(f"Final Fidelity: {fidelity}")
