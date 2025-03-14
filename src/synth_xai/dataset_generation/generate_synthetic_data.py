import argparse
import os

default_n_threads = 16
os.environ["OPENBLAS_NUM_THREADS"] = f"{default_n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{default_n_threads}"
os.environ["OMP_NUM_THREADS"] = f"{default_n_threads}"


import signal
import sys
import time
from pathlib import Path
from types import FrameType

import dill
import pandas as pd
import wandb
from loguru import logger
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from synthcity.metrics.eval_detection import (
    SyntheticDetectionGMM,
    SyntheticDetectionLinear,
    SyntheticDetectionMLP,
    SyntheticDetectionXGB,
)
from synthcity.metrics.eval_performance import (
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
)
from synthcity.metrics.eval_privacy import (
    DeltaPresence,
    IdentifiabilityScore,
    kAnonymization,
    kMap,
    lDiversityDistinct,
)
from synthcity.metrics.eval_sanity import (
    CloseValuesProbability,
    CommonRowsProportion,
    DataMismatchScore,
    DistantValuesProbability,
    NearestSyntheticNeighborDistance,
)
from synthcity.metrics.eval_statistical import (
    ChiSquaredTest,
    InverseKLDivergence,
    JensenShannonDistance,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    WassersteinDistance,
)
from synthcity.plugins.core.dataloader import GenericDataLoader

from synth_xai.utils import (
    prepare_adult,
    prepare_breast_cancer,
    prepare_covertype,
    prepare_diabetes,
    prepare_dutch,
    prepare_house16,
    prepare_letter,
    prepare_pima,
    prepare_shuttle,
)

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--dataset_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--samples_to_generate", type=int, nargs="+", default=None)
parser.add_argument("--epochs", type=int, nargs="+", default=300)
parser.add_argument("--synthesizer_name", type=str, default="ctgan")
parser.add_argument("--store_path", type=str, required=True)


def signal_handler(_sig: int, frame: FrameType | None) -> None:
    """
    Function used to handle the SIGINT signal
    """
    logger.info("Gracefully stopping your experiment! Keep calm!")
    global wandb_run
    if wandb_run:
        wandb_run.finish()
    sys.exit(0)


def eval_sanity(metrics: dict, original_dataloader: GenericDataLoader, synth_dataloader: GenericDataLoader) -> None:
    """
    Function used to evaluate the sanity of the synthetic data
    Sanity Metrics that are computed:
        - data_mismatch
        - common_rows_proportion
        - nearest_syn_neighbor_distance
        - close_values_probability
        - distant_values_probability

    Params:
        metrics: The metrics dictionary where the results will be stored
        original_dataloader: The original data loader
        synth_dataloader: The synthetic data loader

    Returns:
        None

    """
    metrics["sanity"] = {}
    try:
        data_mismatch = DataMismatchScore()
        metrics["sanity"]["data_mismatch"] = data_mismatch.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)[
            "score"
        ]
    except Exception as e:
        logger.error(f"Error computing data_mismatch: {e}")

    try:
        dvp = DistantValuesProbability()
        metrics["sanity"]["distant_values_probability"] = dvp.evaluate(
            X_gt=original_dataloader, X_syn=synth_dataloader
        )["score"]
    except Exception as e:
        logger.error(f"Error computing distant_values_probability: {e}")

    try:
        nsnd = NearestSyntheticNeighborDistance()
        metrics["sanity"]["nearest_synthetic_neighbor_distance"] = nsnd.evaluate(
            X_gt=original_dataloader, X_syn=synth_dataloader
        )["mean"]
    except Exception as e:
        logger.error(f"Error computing nearest_synthetic_neighbor_distance: {e}")

    try:
        cvp = CloseValuesProbability()
        metrics["sanity"]["close_values_probability"] = cvp.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)[
            "score"
        ]
    except Exception as e:
        logger.error(f"Error computing close_values_probability: {e}")

    try:
        crp = CommonRowsProportion()
        metrics["sanity"]["common_rows_proportion"] = crp.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)[
            "score"
        ]
    except Exception as e:
        logger.error(f"Error computing common_rows_proportion: {e}")


def eval_statistical(
    metrics: dict, original_dataloader: GenericDataLoader, synth_dataloader: GenericDataLoader
) -> None:
    """
    Function to compute the statistical Metrics of the synthetic data:
        - inverse_kl_divergence
        - ks_test
        - chi_squared_test
        - max_mean_discrepancy
        - jensenshannon_dist
        - wasserstein_dist

    Params:
        metrics: The metrics dictionary where the results will be stored
        original_dataloader: The original data loader
        synth_dataloader: The synthetic data loader

    Returns:
        None

    """
    metrics["statistical"] = {}

    try:
        chi_squared_test = ChiSquaredTest()
        metrics["statistical"]["chi_squared_test"] = chi_squared_test.evaluate(
            X_gt=original_dataloader, X_syn=synth_dataloader
        )["marginal"]
    except Exception as e:
        logger.error(f"Error computing chi_squared_test: {e}")

    try:
        inverse_kl_divergence = InverseKLDivergence()
        metrics["statistical"]["inverse_kl_divergence"] = inverse_kl_divergence.evaluate(
            X_gt=original_dataloader, X_syn=synth_dataloader
        )["marginal"]
    except Exception as e:
        logger.error(f"Error computing inverse_kl_divergence: {e}")

    try:
        ks_test = KolmogorovSmirnovTest()
        metrics["statistical"]["ks_test"] = ks_test.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)[
            "marginal"
        ]
    except Exception as e:
        logger.error(f"Error computing ks_test: {e}")

    try:
        max_mean_discrepancy = MaximumMeanDiscrepancy()
        metrics["statistical"]["max_mean_discrepancy"] = max_mean_discrepancy.evaluate(
            X_gt=original_dataloader, X_syn=synth_dataloader
        )["joint"]
    except Exception as e:
        logger.error(f"Error computing max_mean_discrepancy: {e}")

    try:
        jensenshannon_dist = JensenShannonDistance()
        metrics["statistical"]["jensenshannon_dist"] = jensenshannon_dist.evaluate(
            X_gt=original_dataloader, X_syn=synth_dataloader
        )["marginal"]
    except Exception as e:
        logger.error(f"Error computing jensenshannon_dist: {e}")

    try:
        wasserstein_dist = WassersteinDistance()
        metrics["statistical"]["wasserstein_dist"] = wasserstein_dist.evaluate(
            X_gt=original_dataloader, X_syn=synth_dataloader
        )["joint"]
    except Exception as e:
        logger.error(f"Error computing wasserstein_dist: {e}")


def evaluate_quality(
    metrics: dict, original_dataloader: GenericDataLoader, synth_dataloader: GenericDataLoader
) -> None:
    """
    Function to compute the Synthetic Data quality metrics:
        - performance.xgb
        - performance.linear
        - performance.mlp
        - performance.feat_rank_distance

    Params:
        metrics: The metrics dictionary where the results will be stored
        original_dataloader: The original data loader
        synth_dataloader: The synthetic data loader

    Returns:
        None

    """
    metrics["quality"] = {}

    try:
        xgb = PerformanceEvaluatorXGB()
        res = xgb.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)
        metrics["quality"]["xgb_gt"] = res["gt"]
        metrics["quality"]["xgb_syn_id"] = res["syn_id"]
        metrics["quality"]["xgb_syn_ood"] = res["syn_ood"]
    except Exception as e:
        logger.error(f"Error computing xgb performance: {e}")

    try:
        linear = PerformanceEvaluatorLinear()
        res = linear.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)
        metrics["linear_gt"] = res["gt"]
        metrics["linear_syn_id"] = res["syn_id"]
        metrics["linear_syn_ood"] = res["syn_ood"]
    except Exception as e:
        logger.error(f"Error computing linear performance: {e}")

    try:
        mlp = PerformanceEvaluatorMLP()
        res = mlp.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)
        metrics["mlp_gt"] = res["gt"]
        metrics["mlp_syn_id"] = res["syn_id"]
        metrics["mlp_syn_ood"] = res["syn_ood"]
    except Exception as e:
        logger.error(f"Error computing mlp performance: {e}")

    # try:
    #     feat_rank_distance = FeatureImportanceRankDistance()
    #     res = feat_rank_distance.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)
    #     metrics["feat_rank_distance_corr"] = res["corr"]
    #     metrics["feat_rank_distance_pvalue"] = res["pvalue"]
    # except Exception as e:
    #     logger.error(f"Error computing feature importance rank distance: {e}")


def evaluate_detection(
    metrics: dict, original_dataloader: GenericDataLoader, synth_dataloader: GenericDataLoader
) -> None:
    """
    Function to compute the detection metrics on the synthetic data:
        - detection_gmm
        - detection_xgb
        - detection_mlp
        - detection_linear

    Params:
        metrics: The metrics dictionary where the results will be stored
        original_dataloader: The original data loader
        synth_dataloader: The synthetic data loader

    Returns:
        None

    """
    metrics["detection"] = {}

    try:
        detection_gmm = SyntheticDetectionGMM()
        metrics["detection"]["gmm"] = detection_gmm.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)["mean"]
    except Exception as e:
        logger.error(f"Error computing detection_gmm: {e}")

    try:
        detection_xgb = SyntheticDetectionXGB()
        metrics["detection"]["xgb"] = detection_xgb.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)["mean"]
    except Exception as e:
        logger.error(f"Error computing detection_xgb: {e}")

    try:
        detection_mlp = SyntheticDetectionMLP()
        metrics["detection"]["mlp"] = detection_mlp.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)["mean"]
    except Exception as e:
        logger.error(f"Error computing detection_mlp: {e}")

    try:
        detection_linear = SyntheticDetectionLinear()
        metrics["detection"]["linear"] = detection_linear.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)[
            "mean"
        ]
    except Exception as e:
        logger.error(f"Error computing detection_linear: {e}")


def evaluate_privacy(
    metrics: dict, original_dataloader: GenericDataLoader, synth_dataloader: GenericDataLoader
) -> None:
    """
    Function to compute the privacy metrics on the synthetic data:

        - k_anonymization
        - l_diversity
        - kmap
        - delta_presence
        - identifiability_score

    Params:
        metrics: The metrics dictionary where the results will be stored
        original_dataloader: The original data loader
        synth_dataloader: The synthetic data loader

    Returns:
        None

    """
    metrics["privacy"] = {}

    try:
        k_anonymization = kAnonymization()
        res = k_anonymization.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)
        metrics["privacy"]["k_anonymization_gt"] = res["gt"]
        metrics["privacy"]["k_anonymization_syn"] = res["syn"]
    except Exception as e:
        logger.error(f"Error computing k_anonymization: {e}")

    try:
        l_diversity = lDiversityDistinct()
        res = l_diversity.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)
        metrics["privacy"]["l_diversity_gt"] = res["gt"]
        metrics["privacy"]["l_diversity_syn"] = res["syn"]
    except Exception as e:
        logger.error(f"Error computing l_diversity: {e}")

    try:
        kmap = kMap()
        res = kmap.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)
        metrics["privacy"]["kmap"] = res["score"]
    except Exception as e:
        logger.error(f"Error computing kmap: {e}")

    try:
        delta_presence = DeltaPresence()
        res = delta_presence.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)
        metrics["privacy"]["delta_presence_gt"] = res["score"]
    except Exception as e:
        logger.error(f"Error computing delta_presence: {e}")

    try:
        identifiability_score = IdentifiabilityScore()
        res = identifiability_score.evaluate(X_gt=original_dataloader, X_syn=synth_dataloader)
        metrics["privacy"]["identifiability_score_OC"] = res["score_OC"]
        metrics["privacy"]["identifiability_score"] = res["score"]
    except Exception as e:
        logger.error(f"Error computing identifiability_score: {e}")


def compute_metrics(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, outcome_variable: str) -> dict:
    """
    Compute metrics to evaluate the quality of the synthetic data

    Args:
        real_data: The real data
        synthetic_data: The synthetic data
        outcome_variable: The outcome variable of the dataset

    Returns:
        metrics: The metrics computed

    """
    original_dataloader = GenericDataLoader(data=real_data, target_column=outcome_variable)
    synth_dataloader = GenericDataLoader(data=synthetic_data, target_column=outcome_variable)
    metrics: dict[str, dict[str, float | dict[str, float]]] = {}

    eval_sanity(metrics, original_dataloader, synth_dataloader)

    eval_statistical(metrics, original_dataloader, synth_dataloader)

    evaluate_quality(
        metrics=metrics,
        original_dataloader=original_dataloader,
        synth_dataloader=synth_dataloader,
    )

    evaluate_privacy(
        metrics=metrics,
        original_dataloader=original_dataloader,
        synth_dataloader=synth_dataloader,
    )

    evaluate_detection(
        metrics=metrics,
        original_dataloader=original_dataloader,
        synth_dataloader=synth_dataloader,
    )
    return metrics


def get_real_data(dataset_name: str) -> tuple[pd.DataFrame, str]:
    """
    Get the preprocessed real data based on the dataset name. We will use
    these data to train the synthesizer and evaluate the quality of the synthetic data

    Args:
        dataset_name: The name of the dataset to use

    Returns:
        real_data: The preprocessed real data
        outcome_variable: The outcome variable of the dataset

    """
    current_script_path = Path(__file__).resolve().parent
    match dataset_name:
        case "pima":
            _, _, _, _, _, _, real_data, _ = prepare_pima(sweep=False, seed=args.seed, current_path=current_script_path)
            outcome_variable = "Outcome"
        case "adult":
            _, _, _, _, _, _, real_data, _ = prepare_adult(
                sweep=False, seed=args.seed, current_path=current_script_path
            )

            outcome_variable = "income_binary"
        case "breast_cancer":
            _, _, _, _, _, _, real_data, _ = prepare_breast_cancer(
                sweep=False, seed=args.seed, current_path=current_script_path
            )
            outcome_variable = "Status"
        case "diabetes":
            _, _, _, _, _, _, real_data, _ = prepare_diabetes(
                sweep=False, seed=args.seed, current_path=current_script_path
            )
            outcome_variable = "readmitted"
        case "dutch":
            _, _, _, _, _, _, real_data, _ = prepare_dutch(
                sweep=False, seed=args.seed, current_path=current_script_path
            )
            outcome_variable = "occupation_binary"
        case "letter":
            _, _, _, _, _, _, real_data, _ = prepare_letter(sweep=False, seed=args.seed)
            outcome_variable = "letter"
        case "shuttle":
            _, _, _, _, _, _, real_data, _ = prepare_shuttle(sweep=False, seed=args.seed)
            outcome_variable = "class"
        case "covertype":
            _, _, _, _, _, _, real_data, _ = prepare_covertype(sweep=False, seed=args.seed)
            outcome_variable = "cover_type"
        case "house16":
            _, _, _, _, _, _, real_data, _ = prepare_house16(sweep=False, seed=args.seed)
            outcome_variable = "class"
        case _:
            msg = "Invalid dataset name"
            raise ValueError(msg)
    return real_data, outcome_variable


def log_losses(synthesizer: TVAESynthesizer | CTGANSynthesizer, synthesizer_name: str) -> None:
    """
    Log the losses of the synthesizer. This depends on the synthesizer used.
    For instance for the TVAESynthesizer, we only have one loss while for the
    CTGANSynthesizer we have two losses: Generator Loss and Discriminator Loss

    Args:
        synthesizer: The synthesizer used to generate the synthetic data
        synthesizer_name: The name of the synthesizer used

    Returns:
        None

    """
    wandb_run = wandb.init(
        project="tango_generation",
        name=f"{args.synthesizer_name}_epochs_{epochs}",
        config={
            "dataset": args.dataset_name,
            "epochs": epochs,
            "synthesizer_name": args.synthesizer_name,
        },
    )

    losses = synthesizer.get_loss_values()
    match synthesizer_name:
        case "ctgan":
            loss_generator = list(losses["Generator Loss"])
            loss_discriminator = list(losses["Discriminator Loss"])

            for epoch, loss in enumerate(loss_discriminator):
                wandb.log({"loss_discriminator": loss, "epoch": epoch})
            for epoch, loss in enumerate(loss_generator):
                wandb.log({"loss_generator": loss, "epoch": epoch})
        case "tvae":
            loss_gan = list(losses["Loss"])
            for epoch, loss in enumerate(loss_gan):
                wandb.log({"loss": loss, "epoch": epoch})

    wandb_run.finish()


def get_synthesizer(
    synthesizer_name: str, metadata: Metadata, epochs: int, cuda: bool = True
) -> TVAESynthesizer | CTGANSynthesizer:
    """
    Get the synthesizer based on the synthesizer name

    Args:
        synthesizer_name: The name of the synthesizer to use
        metadata: The metadata of the dataset
        epochs: The number of epochs to train the synthesizer
        cuda: Whether to use cuda or not

    Returns:
        synthesizer: The synthesizer to use

    """
    match synthesizer_name:
        case "ctgan":
            return CTGANSynthesizer(metadata, epochs=epochs, cuda=cuda)
        case "tvae":
            return TVAESynthesizer(metadata, epochs=epochs, cuda=cuda)
        case _:
            msg = "Invalid synthesizer name"
            raise ValueError(msg)


def has_object_dtypes(data: pd.DataFrame) -> bool:
    """
    Check if the data has object data types

    Args:
        data: The data to check

    Returns:
        has_object: Whether the data has object data types or not

    """
    return data.select_dtypes(include=["object"]).shape[1] > 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    logger.info("Starting the synthetic data generation process...")

    args = parser.parse_args()

    if args.dataset_name is None or args.samples_to_generate is None:
        msg = "Please provide dataset_name, samples_to_generate"
        raise ValueError(msg)

    logger.info("Loading real data...")
    real_data, outcome_variable = get_real_data(args.dataset_name)

    if has_object_dtypes(real_data):
        real_data = real_data.apply(pd.to_numeric, errors="coerce")

    logger.info("Detecting Metadata")
    metadata = Metadata.detect_from_dataframe(data=real_data, table_name=args.dataset_name)

    logger.info(f"Initializing the {args.synthesizer_name} synthesizer...")
    synthesizers = []
    logger.info(f"Starting fitting the {args.synthesizer_name} synthesizer...")
    possible_epochs = list(args.epochs)
    for epochs in possible_epochs:
        # load the synthesizer from disk if it exists
        if Path(
            f"{args.store_path}{args.dataset_name}/synthesizer/synthesizer_{args.synthesizer_name}_{epochs}.pkl"
        ).exists():
            logger.info(f"Loading the synthesizer with {epochs} epochs...")
            with Path(
                f"{args.store_path}{args.dataset_name}/synthesizer/synthesizer_{args.synthesizer_name}_{epochs}.pkl"
            ).open("rb") as f:
                synthesizer = dill.load(f)
            synthesizers.append((epochs, synthesizer))
        else:
            logger.info(f"Fitting the synthesizer with {epochs} epochs...")
            # We do a sort of hyperparameter tuning by training the synthesizer with different epochs.
            # Apparently there is not a real hyperparameter tuning for the synthesizer, the only thing
            # that we can test is the number of epochs
            # https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/ctgansynthesizer#how-do-i-tune-the-hyperparameters-such-as-epochs-or-other-values
            synthesizer = get_synthesizer(synthesizer_name=args.synthesizer_name, metadata=metadata, epochs=epochs)
            synthesizer.fit(real_data)
            # Each gan is stored on disk to be used later
            with Path(
                f"{args.store_path}{args.dataset_name}/synthesizer/synthesizer_{args.synthesizer_name}_{epochs}.pkl"
            ).open("wb") as f:
                dill.dump(synthesizer, f)
            synthesizers.append((epochs, synthesizer))
            # At the end of the training we can log and plot the losses of the synthesizer
            log_losses(synthesizer, args.synthesizer_name)

    logger.info("Model fitting completed...")

    # measure the time needed to generate the samples
    samples_to_generate = list(args.samples_to_generate)
    print(samples_to_generate, synthesizers)
    for generated_dataset_size in samples_to_generate:
        print("generated_dataset_size", generated_dataset_size)
        for epochs, synthesizer in synthesizers:
            if Path(
                f"{args.store_path}{args.dataset_name}/synthetic_data/synthetic_data_{generated_dataset_size}_epochs_{epochs}_synthethizer_name_{args.synthesizer_name}.csv"
            ).exists():
                logger.info(f"Loading synthetic data with {generated_dataset_size} samples...")
                synthetic_data = pd.read_csv(
                    Path(
                        f"{args.store_path}{args.dataset_name}/synthetic_data/synthetic_data_{generated_dataset_size}_epochs_{epochs}_synthethizer_name_{args.synthesizer_name}.csv"
                    )
                )
                logger.info(f"Loaded synthetic data with {generated_dataset_size} samples...")

                wandb_run = wandb.init(
                    project="tango_generation",
                    name=f"generated_dataset_size_{generated_dataset_size}_epochs_{epochs}",
                    config={
                        "dataset": args.dataset_name,
                        "samples_to_generate": generated_dataset_size,
                        "epochs": epochs,
                        "synthesizer": args.synthesizer_name,
                    },
                )

                logger.info("Evaluating the quality of the synthetic data...")

                # evaluate quality
                metrics = compute_metrics(real_data, synthetic_data, outcome_variable)
                logger.info("Evaluated the quality")

                wandb_run.log(metrics["sanity"])
                wandb_run.log(metrics["statistical"])
                wandb_run.log(metrics["quality"])
                wandb_run.log(metrics["detection"])
                wandb_run.log(metrics["privacy"])

                wandb_run.finish()
            else:
                logger.info(f"Generating {generated_dataset_size} samples for the {args.dataset_name} dataset...")

                start = time.time()
                synthetic_data = synthesizer.sample(num_rows=generated_dataset_size)

                if has_object_dtypes(synthetic_data):
                    synthetic_data = synthetic_data.apply(pd.to_numeric, errors="coerce")

                end_time = time.time()
                elapsed_time = end_time - start

                # save synthetic data
                synthetic_data.to_csv(
                    Path(
                        f"{args.store_path}{args.dataset_name}/synthetic_data/synthetic_data_{generated_dataset_size}_epochs_{epochs}_synthethizer_name_{args.synthesizer_name}.csv"
                    ),
                    index=False,
                )

            wandb_run = wandb.init(
                project="tango_generation",
                name=f"generated_dataset_size_{generated_dataset_size}_epochs_{epochs}",
                config={
                    "dataset": args.dataset_name,
                    "samples_to_generate": generated_dataset_size,
                    "epochs": epochs,
                    "synthesizer": args.synthesizer_name,
                },
            )

            logger.info("Evaluating the quality of the synthetic data...")

            # evaluate quality
            metrics = compute_metrics(real_data, synthetic_data, outcome_variable)
            wandb_run.log({"generation_time": elapsed_time})
            wandb_run.log(metrics["sanity"])
            wandb_run.log(metrics["statistical"])
            wandb_run.log(metrics["quality"])
            wandb_run.log(metrics["detection"])
            wandb_run.log(metrics["privacy"])

            wandb_run.finish()
