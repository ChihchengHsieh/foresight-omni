import json
import math
import os, logging, warnings, torch, argparse, datetime, random, time, gc
import pandas as pd
import numpy as np
from evaluators.temporal import TemporalDiseaseEvaluator, TemporalEvalConfig
import utils.misc as utils
from model.universal_dense_vit_improved import build_universal_dense_vit
from utils.sampler import ImbalancedDatasetSampler, MultiLabelImbalancedDatasetSampler
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from model.builder import setup_distributed_model
from utils.checkpoint import (
    save_dist_with_time,
    load_checkpoint_from_path,
)
from dataset.universal_public_dataset import build_combined_datasets
from dataset.universal_image import build_image_level_universal_datasets
from dataset.builder import build_loader
from utils.checkpoint import load_continue_training, get_trained_epoch_from_name
from engine.universal_dense_onehead import (
    train_one_epoch,
    evaluate,
)
from utils.logger import GeneralTrainingLogger
import torch.distributed as dist
from utils.parameters import initialize_weights, print_parameters_count
from model.universal_dense.input_output_projs import (
    build_universal_input_output_projs_batch,
)
from copy import deepcopy
from torch.distributed.elastic.multiprocessing.errors import record
from model.universal_dense.patch_emb import build_patch_embedding
from utils.gpu import log_gpu_info
from collections import OrderedDict
import torch.nn as nn
from evaluators.classification import ClassificationEvaluator
from evaluators.mse import MSEEvaluator
from utils.args import list_of_floats, list_of_ints, list_of_str
from utils.warning import supress_warnings
from transformers import get_cosine_schedule_with_warmup
from dataset.aug import get_default_aug
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from dataset.clsa_muiltimodal import CLSAMultimodalDataset
from dataset.participant_split import assign_participant_finetune_split
from training.model_selection import (
    collect_auroc_values,
    get_model_selection_value,
    update_patience_counter,
)

supress_warnings()


def only_keep_cases_with(df, col):
    # only keep the cases where the col is not None
    return df[col].notna() & (df[col] != "")


def clean_log_record(record):
    return {
        k: v
        for k, v in record.items()
        if not (isinstance(v, float) and np.isnan(v))
    }


def cumulative_selection_labels(args):
    """Match UKB checkpoint selection to the cumulative disease heads only."""
    return [
        f"={disease}_{year}"
        for disease in args.diseases
        for year in args.progression_label_years
    ]


def load_logger_history(logger, finetune_output_dir, resume_epoch):
    train_log_path = os.path.join(finetune_output_dir, "train_logs.csv")
    val_log_path = os.path.join(finetune_output_dir, "val_logs.csv")

    if os.path.exists(train_log_path):
        train_df = pd.read_csv(train_log_path, index_col=0)
        train_df = train_df.iloc[:resume_epoch]
        logger.train_logs = [
            clean_log_record(record) for record in train_df.to_dict("records")
        ]
        logging.info(
            f"Loaded {len(logger.train_logs)} previous CLSA finetune train logs."
        )

    if logger.inspecting and os.path.exists(val_log_path):
        val_df = pd.read_csv(val_log_path, index_col=0)
        val_df = val_df.iloc[:resume_epoch]
        logger.val_logs = [
            clean_log_record(record) for record in val_df.to_dict("records")
        ]
        logging.info(
            f"Loaded {len(logger.val_logs)} previous CLSA finetune validation logs."
        )


def checkpoint_epoch(path):
    return get_trained_epoch_from_name(os.path.basename(path))


def get_existing_best_checkpoints(output_dir, affix, resume_epoch, buffer_size):
    paths = [
        str(p)
        for p in Path(output_dir).glob(f"epoch_*_{affix}model")
        if checkpoint_epoch(str(p)) < resume_epoch
    ]
    paths.sort(key=checkpoint_epoch)
    return paths[-buffer_size:]


def restore_finetune_best_state(args, output_dir, finetune_output_dir, resume_epoch):
    val_log_path = os.path.join(finetune_output_dir, "val_logs.csv")
    if os.path.exists(val_log_path):
        val_df = pd.read_csv(val_log_path, index_col=0).iloc[:resume_epoch]
        if "loss" in val_df:
            finite_losses = pd.to_numeric(val_df["loss"], errors="coerce").dropna()
            if len(finite_losses) > 0:
                main.best_loss = float(finite_losses.min())

        supported_means = []
        for record in val_df.to_dict("records"):
            supported_aurocs = collect_auroc_values(
                record,
                min_positive_samples=10,
                selected_labels=cumulative_selection_labels(args),
            )
            supported_aurocs = [
                value for value in supported_aurocs if np.isfinite(value)
            ]
            supported_means.append(
                float(np.mean(supported_aurocs)) if supported_aurocs else np.nan
            )
        finite_supported_means = pd.Series(supported_means).dropna()
        if len(finite_supported_means) > 0:
            main.best_auroc = float(finite_supported_means.max())

    main.best_loss_buffer = get_existing_best_checkpoints(
        output_dir,
        "best_finetune_loss",
        resume_epoch,
        args.best_model_buffer_size,
    )
    main.best_auroc_buffer = get_existing_best_checkpoints(
        output_dir,
        "best_finetune_auroc",
        resume_epoch,
        args.best_model_buffer_size,
    )
    logging.info(
        "Restored finetune best state: "
        f"best_loss={main.best_loss}, best_auroc={main.best_auroc}, "
        f"loss_buffer={len(main.best_loss_buffer)}, "
        f"auroc_buffer={len(main.best_auroc_buffer)}"
    )


# fmt: off
def get_args_parser():
    parser = argparse.ArgumentParser("Detection Training Script", add_help=False)

    # Dataset
    parser.add_argument("--image_size", default=None, type=int, help="size of the images used in training, validation and testing.")

    # Files and Paths
    parser.add_argument("--name", default="test", type=str, help="Name of the model")
    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")

    # Training
    parser.add_argument("--val_freq", default=1, type=int, help="Frequency of epochs to run evaluation, set it to 1 for running after every epoch.")
    parser.add_argument("--verbose", action="store_true", help="Print the info and debug information.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--imbalanced", action="store_true", help="if true, the dataset provide is imbalanced, and imbalanced sampler will be applied.")
    parser.add_argument("--num_samples", default=None, type=int, help="Instance to pass through an imbalanced dataset for an epoch.")
    parser.add_argument("--sharding", action="store_true", help="Shard the model.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--inspecting", action="store_true")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_scheduler", action="store_true", help="Whether to use learning rate scheduler.")
    parser.add_argument("--lr_drop", default=None, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--prefetch_factor", default=2, type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--clip_max_norm", default=1, type=float, help="gradient clipping max norm")
    parser.add_argument("--random_sample_start_epoch", default=None, type=int, help="The epoch to start random sampling ")
    parser.add_argument("--sample_val_ratio", default=None, type=float, help="Ratio of the training dataset set used for validation. (The instances are sampled from validation dataset.)")
    parser.add_argument("--elastic", action="store_true", help="Use L1 regularization with coef as weight_decay")
    parser.add_argument("--class_weight", action="store_true", help="Weight for positive instances in the binary cross entropy loss.")
    parser.add_argument("--pos_weight_scale", default=1, type=float, help="Scale the positive weight in the binary cross entropy loss.")
    parser.add_argument("--binary_loss_type", default="bce", type=str, help="Loss for binary classification tasks.")
    parser.add_argument("--early_stop", action="store_true", help="Whether to use early stopping.")
    parser.add_argument("--glaucoma_side_check", action="store_true")
    parser.add_argument("--see_no_side_as_both", action="store_true")
    parser.add_argument("--no_quality_control", action="store_true"),
    parser.add_argument("--include_self_report_label", action="store_true")
    parser.add_argument("--include_eye_problem_label", action="store_true")
    parser.add_argument("--survival_analysis_label", action="store_true", help="Whether to use survival analysis for the progression label.")
    parser.add_argument("--no_aug", action="store_true", help="Not using augmentation during training.")
    parser.add_argument("--enhanced_aug", action="store_true", help="Whether to use enhanced augmentation.")
    parser.add_argument("--run_xai", action="store_true", help="Whether to run gradcam during validation")
    parser.add_argument("--run_xai_fix", action="store_true", help="Whether to run gradcam for certain indexes during validation")
    parser.add_argument("--ukb_downsampling", action="store_true", help="Whether to down sample the UKB dataset.")
    parser.add_argument("--progression_label_ignorant_label_years", default=1, type=float, help="The years to ignore the progression label.")
    parser.add_argument(
        "--interval_score_mode",
        default="end_horizon",
        choices=["end_horizon", "conditional_risk"],
        help="Score used for temporal interval metrics: end_horizon preserves existing behavior; conditional_risk uses max(P_end - P_start, 0) / max(1 - P_start, eps).",
    )

    # Early stopping
    parser.add_argument("--xai_top_n", default=3, type=int, help="Top n classes to run gradcam on.")
    parser.add_argument("--best_model_buffer_size", default=5, type=int, help="Number of best checkpoints to keep per criterion (loss, auroc) before evicting the oldest.")
    parser.add_argument("--finetune_patience", default=20, type=int, help="Early-stopping patience for CLSA finetuning validation checks. Set negative to disable.")
    parser.add_argument(
        "--reset_finetune_state",
        action="store_true",
        help="Start CLSA adaptation at epoch 0 with a fresh optimizer and scheduler while retaining the loaded model weights.",
    )

    # Model
    parser.add_argument("--continue_training", type=str, default=None,  help="continue to train a model with given path, these weights will replace the pretrained one.")
    parser.add_argument("--pretrained_path", type=str, default=None,  help="path to pretrained weights.")

    ## Transformer
    parser.add_argument("--causal", action="store_true", help="Use casual mask for attention if true")
    parser.add_argument("--attn_dropout_p", default=0.0, type=float, help="Dropout rate for the attention score.",)
    parser.add_argument("--ff_dropout_p", default=0.0, type=float, help="Dropout rate for the attention score.",)
    parser.add_argument("--patch_size", default=16, type=int, help="Size of patches used in input projector.",)
    parser.add_argument("--n_layers", default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument("--dim", default=512, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument("--n_heads", default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument("--n_kv_heads", default=None, type=int, help="Number of n_kv_head for group attention.")
    parser.add_argument("--container", default="bracket", type=str, help="Method to contain modalities.")
    parser.add_argument("--pos", default="sin-input", type=str, help="Positional Encoding strategy.")

    # Genotype
    parser.add_argument("--genotype_autoencoder_path", default=None, type=str, help="Path to the pretrained Genotype Autoencoder")
    parser.add_argument("--genotype_autoencoder_intermediate_dims", default="32,2048", type=list_of_ints, help="Intermediate dimensions for the genotype autoencoder.")
    parser.add_argument("--genotype_autoencoder_patch_sizes", default="64,64", type=list_of_ints, help="Patch sizes for the genotype autoencoder.")
    parser.add_argument("--genotype_last_patch_size", default="64", type=list_of_ints, help="Intermediate dimensions for the genotype embedding.")
    parser.add_argument("--genotype_length", default=15889070, type=int, help="Length of the genotype.")
    parser.add_argument("--genotype_lifetime_prediction", action="store_true", help="Whether to use genotype for predicting glaucoma progression.")
    # use combined dataset

    # Multimodal controls
    parser.add_argument("--input_modalities", default=["fundus_image"], type=list_of_str, help="The input modalities to use. (fundus_image, prs, genotype, anthropometrics, family_history, principal_components, lifestyle, mental_health, socioeconomic, vitals, medications)")
    parser.add_argument("--omics_num_tokens", default=8, type=int, help="Number of tokens emitted per omics modality.")
    parser.add_argument("--omics_tokenizer", default="grouped", choices=["grouped", "linear"], help="Omics tokenizer: grouped is missingness-aware; linear preserves the legacy projection.")
    parser.add_argument("--omics_dropout_p", default=0.0, type=float, help="Dropout used inside the grouped omics tokenizer.")
    parser.add_argument("--no_omics_qc", dest="omics_qc", action="store_false", help="Disable train-fitted omics QC.")
    parser.set_defaults(omics_qc=True)
    parser.add_argument("--omics_missing_rate_threshold", default=0.40, type=float, help="Mask omics features with train missing rate above this threshold.")
    parser.add_argument("--omics_min_std", default=1e-6, type=float, help="Mask omics features with train standard deviation below this value.")
    parser.add_argument("--omics_min_unique_values", default=10, type=int, help="Mask omics features with fewer train unique non-missing values.")
    parser.add_argument("--omics_winsor_lower_quantile", default=0.005, type=float, help="Train quantile used for lower omics winsorization.")
    parser.add_argument("--omics_winsor_upper_quantile", default=0.995, type=float, help="Train quantile used for upper omics winsorization.")
    parser.add_argument("--progression_label_years", default=[0, 2, 5, 10, 15], type=list_of_ints, help="Years to use for the progression label.")
    # For input modalities, we have:
    parser.add_argument("--diseases", default=[], type=list_of_str, help="Diseases to use for the progression label. Available diseases: glaucoma, ad, pd, hd, ms, cvd,t2d")
    parser.add_argument("--clinical_cat_modalities", default=["instance_comparative_body_size_at_age_10"], type=list_of_str, help="Clinical categorical modalities to use as input.")
    parser.add_argument("--clinical_num_modalities", default=["instance_age_at_time", "instance_iop", "instance_systolic_bp", "instance_diastolic_bp", "instance_pulse_rate", "instance_height_cm", "instance_weight_kg", "instance_body_mass_index_bmi", "instance_waist_circumference_cm", "instance_hip_circumference_cm", ], type=list_of_str, help="Clinical numerical modalities to use as input.")
    # For labels, we have:
    parser.add_argument("--cat_labels", default=None, type=list_of_str, help="Categorical labels to use. If None, use all available categorical labels.")
    parser.add_argument("--numerical_labels", default=None, type=list_of_str, help="Numerical labels to use. If None, use all available numerical labels.")
    # dropout params:
    parser.add_argument("--fundus_sample_prob", default=1.0, type=float, help="Sample probability for fundus images during training.")
    # finetune_epochs
    parser.add_argument("--finetune_epochs", default=50, type=int, help="Number of epochs to finetune on CLSA dataset.")
    parser.add_argument("--finetune_portions", default=[0.4], type=list_of_floats, help="Comma-separated CLSA finetuning data fractions, e.g. 0.4 or 0.2,0.4,0.6.")
    parser.add_argument("--finetune_train_frac", default=0.75, type=float, help="Fraction of each CLSA finetune subset used for finetune train split (rest is validation).")
    parser.add_argument(
        "--clsa_processed_df_path",
        default=os.environ.get(
            "FORESIGHT_OMNI_CLSA_MANIFEST", "data/clsa/clsa_multimodal.parquet"
        ),
        type=str,
        help="Prepared CLSA multimodal manifest used for direct evaluation and adaptation.",
    )
    parser.add_argument(
        "--ukb_mean_std_map_path",
        default=os.environ.get(
            "FORESIGHT_OMNI_UKB_MEAN_STD", "data/ukb/ukbb_mean_std_map.json"
        ),
        type=str,
        help="Training-cohort normalization statistics saved by the UKB run.",
    )
    parser.add_argument("--do_clsa_finetune", dest="do_clsa_finetune", action="store_true", help="Run CLSA finetuning after UKB evaluation.")
    parser.add_argument("--no_clsa_finetune", dest="do_clsa_finetune", action="store_false", help="Skip CLSA finetuning after UKB evaluation.")
    parser.set_defaults(do_clsa_finetune=True)
    parser.add_argument("--eval_only_finetune_checkpoint", action="store_true", help="Build the CLSA finetune split and evaluate the loaded checkpoint without running finetuning.")
    parser.add_argument("--eval_only_finetune_split", default="test", choices=["val", "test"], help="CLSA finetune split to evaluate when --eval_only_finetune_checkpoint is set.")

    # stage 1 model path
    parser.add_argument("--stage1_model_path", default=None, type=str, help="Path to the stage 1 trained model.")

    # new arguments
    parser.add_argument("--normalise_fundus_image", action="store_true", help="Whether to normalise fundus images.")
    parser.add_argument("--use_combined_dataset", action="store_true", help="Whether to use the combined CLSA dataset.")
    parser.add_argument("--image_encoder_type", default="patch_emb", type=str, help="Type of image encoder to use. Options are 'patch_emb' or 'cnn'.")

    # algo_qc
    parser.add_argument("--algo_qc", action="store_true", help="Whether to use algorithmic quality control for fundus images.")
    parser.add_argument(
        "--clsa_qc_strategy",
        default="legacy",
        choices=["legacy", "algorithmic_only"],
        help="CLSA image-QC implementation. algorithmic_only requires an explicit path-specific algorithmic label for every retained image.",
    )
    parser.add_argument(
        "--clsa_direct_eval_scope",
        default="full",
        choices=["full", "locked_test"],
        help="Evaluate the fixed UKB checkpoint on the full CLSA cohort or on the participant-disjoint locked test split.",
    )

    # weight_sample_validation
    parser.add_argument("--weight_sample_validation", action="store_true", help="Whether to use weighted sampling for validation dataset.")

    return parser
# fmt: on


@record
def main(args):
    ##################################################
    #   Initialisation
    ##################################################

    if args.debug:
        # force verbose in debugging mode.
        args.verbose = True

    logging.basicConfig(
        filename=None if args.debug else (args.output_dir + f"{args.name}.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",  # Includes timestamp
        datefmt="%Y-%m-%d %H:%M:%S",  # Custom time format (optional)
    )

    ### Log the args
    logging.info(
        "============================= Arguments ============================="
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info(
        "======================================================================"
    )

    logging.info("Initialisation...")
    utils.init_distributed_mode(args)
    output_dir = Path(args.output_dir)

    # clean up gpu memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logging.info(torch.cuda.memory_summary())

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #################################
    #   Setting Modalities & Labels
    #################################

    diseases = args.diseases
    restricted_inputs = args.input_modalities
    progression_label_years = args.progression_label_years
    all_plotting_names = diseases

    if args.input_modalities is None or len(args.input_modalities) == 0:
        raise ValueError(
            "No input modalities provided, please provide at least one input modality."
        )
    # Creating labels
    cat_labels = []
    if args.cat_labels is not None and len(args.cat_labels) > 0:
        # use only the provided cat labels
        cat_labels.extend(args.cat_labels)

    if args.diseases is not None and len(args.diseases) > 0:
        disease_cat_labels = [
            f"has_{disease}_in_{y}_years"
            for disease in diseases
            for y in progression_label_years
        ]
        cat_labels.extend(disease_cat_labels)

    if args.numerical_labels is not None and len(args.numerical_labels) > 0:
        # use only the provided numerical labels
        numerical_labels = args.numerical_labels
    else:
        # numerical_labels = [f"instance_{d}_after_days" for d in diseases]
        numerical_labels = []

    possible_labels = cat_labels + numerical_labels
    balance_label_cols = possible_labels

    label_num_classes = {d: len(args.progression_label_years) for d in diseases}
    label_tokens_len = {d: 1 for d in diseases}

    # For the labels define evaluators and losses
    eval_inputs = restricted_inputs
    eval_labels = possible_labels

    ### Log the input and output modalities
    logging.info(f"Input Modalities: {restricted_inputs}")
    logging.info(f"Categorical Labels: {cat_labels}")
    logging.info(f"Numerical Labels: {numerical_labels}")
    logging.info(f"Possible Labels: {possible_labels}")

    ##################################################
    #   Dataset configuration
    ##################################################
    logging.info("Dataset configuring")

    # dataset_builder = (
    #     build_combined_datasets
    #     if args.use_combined_dataset
    #     else build_image_level_universal_datasets
    # )

    # train_dataset, val_dataset, test_dataset = dataset_builder(
    #     args,
    #     possible_inputs=restricted_inputs,
    #     possible_labels=possible_labels,
    #     balance_label_cols=balance_label_cols,
    #     quality_control=not args.no_quality_control,
    #     progression_label_years=progression_label_years,
    #     clinical_numerical_features=args.clinical_num_modalities,
    #     clinical_categorical_features=args.clinical_cat_modalities,
    #     enhanced_aug=args.enhanced_aug,
    #     no_aug=args.no_aug,
    #     numerical_label_cols=numerical_labels,  # remember to migrate this to other training and testing files for predicting days.
    #     progression_label_ignorant_label_years=args.progression_label_ignorant_label_years,
    #     normalise_fundus_image=args.normalise_fundus_image,
    #     algo_qc=args.algo_qc,
    #     # diseases=args.diseases,
    # )

    # train_d, val_d, test_d = build_loader(
    #     args,
    #     train_dataset,
    #     val_dataset,
    #     test_dataset,
    #     collate_fn=lambda x: list(x),
    #     weight_sample_validation=args.weight_sample_validation,
    # )

    processed_df_saved_path = args.clsa_processed_df_path
    # check if the processed df exists
    if os.path.exists(processed_df_saved_path):
        logging.info(f"Loading processed df from {processed_df_saved_path}")
        processed_df = pd.read_parquet(processed_df_saved_path)
    else:
        processed_df = None

    with open(Path(args.ukb_mean_std_map_path), "r") as f:
        ukb_mean_std_map = json.load(f)

    # ukb_mean_std_map = deepcopy(train_dataset.mean_std_map)

    if args.clsa_direct_eval_scope == "locked_test":
        if not args.do_clsa_finetune or len(args.finetune_portions) != 1:
            raise ValueError(
                "--clsa_direct_eval_scope locked_test requires CLSA fine-tuning "
                "with exactly one --finetune_portions value."
            )
        processed_df, direct_manifest, direct_split_summary = (
            assign_participant_finetune_split(
                processed_df,
                fine_tune_portion=args.finetune_portions[0],
                train_fraction=args.finetune_train_frac,
                seed=args.seed,
                participant_col="entity_id",
            )
        )
        direct_manifest.to_csv(
            os.path.join(output_dir, "clsa_direct_participant_split_manifest.csv"),
            index=False,
        )
        with open(
            os.path.join(output_dir, "clsa_direct_participant_split_summary.json"),
            "w",
        ) as direct_summary_file:
            json.dump(direct_split_summary, direct_summary_file, indent=2)
        logging.info(
            "CLSA direct evaluation uses the locked participant test split: %s",
            direct_split_summary,
        )

    clsa_dataset = CLSAMultimodalDataset(
        transform=get_default_aug(image_size=args.image_size, split="test"),
        split="test" if args.clsa_direct_eval_scope == "locked_test" else None,
        progression_label_years=progression_label_years,
        fine_tuning=args.clsa_direct_eval_scope == "locked_test",
        progression_label_ignorant_label_years=args.progression_label_ignorant_label_years,
        possible_inputs=restricted_inputs,
        possible_labels=possible_labels,
        mean_std_map=ukb_mean_std_map,
        algo_qc=args.algo_qc,
        clsa_qc_strategy=args.clsa_qc_strategy,
        qc=not args.no_quality_control,
        processed_df=processed_df,
        preprocesed_df_path=processed_df_saved_path,
    )

    # try to load the processed_df again.
    if os.path.exists(processed_df_saved_path):
        logging.info(f"Loading processed df from {processed_df_saved_path}")
        processed_df = pd.read_parquet(processed_df_saved_path)
    else:
        processed_df = None

    clsa_loader_kwargs = {
        "batch_size": args.batch_size,
        "sampler": torch.utils.data.SequentialSampler(clsa_dataset),
        "shuffle": False,
        "collate_fn": lambda x: list(x),
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        clsa_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        clsa_loader_kwargs["persistent_workers"] = args.persistent_workers
    clsa_d = DataLoader(
        clsa_dataset,
        **clsa_loader_kwargs,
    )

    ##################################################
    #   Model Configuration
    ##################################################
    logging.info("Model Configuring...")
    log_gpu_info()

    device = torch.device(args.device, args.local_rank if args.distributed else 0)
    input_to_seq, activations = build_universal_input_output_projs_batch(
        args,
        device,
        possible_input_modalities=restricted_inputs,
        binary_label_cols=possible_labels,
        numerical_label_cols=numerical_labels,
        genotype_autoencoder_intermediate_dims=args.genotype_autoencoder_intermediate_dims,
        genotype_autoencoder_patch_sizes=args.genotype_autoencoder_patch_sizes,
        genotype_last_patch_size=args.genotype_last_patch_size,
        genotype_length=args.genotype_length,
        image_encoder_type=args.image_encoder_type,
        omics_num_tokens=args.omics_num_tokens,
    )
    model = build_universal_dense_vit(
        args,
        input_to_seq=input_to_seq,
        label_num_classes=label_num_classes,
        label_tokens_len=label_tokens_len,
    )
    model.apply(initialize_weights)
    logging.info(model)
    log_gpu_info()

    # whether to load the model
    n_trainable_parameters, n_total_parameters = print_parameters_count(model)
    model.to(device)

    current_epoch = 0
    resume_checkpoint = None

    if args.continue_training:
        cp = load_checkpoint_from_path(args.continue_training, device)
        fixed_state_dict = {}
        for k, v in cp["model"].items():
            new_k = k.replace("fundus-image", "fundus_image")
            fixed_state_dict[new_k] = v

        # raise ValueError("CP contains: ", cp['model'].keys())

        model.load_state_dict(fixed_state_dict, strict=True)

        loaded_epoch = cp.get(
            "epoch",
            get_trained_epoch_from_name(os.path.basename(args.continue_training)),
        )
        current_epoch = loaded_epoch + 1

        resume_checkpoint = {
            k: v for k, v in cp.items() if k in {"optimizer", "scheduler", "epoch"}
        }
        del cp
        del fixed_state_dict
        logging.info(
            f"Continue training state is loaded from [{args.continue_training}], resuming from epoch index {current_epoch}"
        )
        torch.cuda.empty_cache()
        gc.collect()

    model = setup_distributed_model(args, model)

    ##################################################
    #   Training Components Configuration.
    ##################################################
    logging.info("Training Components Configuring...")
    # optimizer = torch.optim.AdamW(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    # )

    # lr_scheduler = None
    # if args.lr_scheduler:
    #     steps_per_epoch = (
    #         len(train_d)
    #         if args.num_samples is None
    #         else args.num_samples // args.batch_size
    #     )
    #     grad_accum = 1
    #     # total optimizer steps seen by scheduler:
    #     num_update_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    #     epoch_to_run = args.epochs - current_epoch + 1
    #     assert (
    #         epoch_to_run > 0
    #     ), f"Training is already done. Current epoch: {current_epoch}, total epochs: {args.epochs}"
    #     num_training_steps = args.epochs * num_update_steps_per_epoch
    #     warmup_ratio = 0.03
    #     num_warmup_steps = int(warmup_ratio * num_training_steps)
    #     lr_scheduler = get_cosine_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=num_training_steps,  # full horizon
    #         num_cycles=0.5,  # monotonic down (no restarts)
    #     )

    ##################################################
    #  Loss & Evaluator Configuration
    ##################################################

    logging.info("Loss and Evaluator Configuring...")
    loss_weight_dict = {l: 1 for l in possible_labels}

    criterions = {}

    # ---------------------------------------------------------
    # 1. NORMAL CATEGORICAL LABELS (keep BCE)
    # ---------------------------------------------------------

    for l in cat_labels:
        criterions[l] = nn.BCEWithLogitsLoss().to(device)

    # ---------------------------------------------------------
    # 2. NUMERICAL LABELS (unchanged)
    # ---------------------------------------------------------
    for l in numerical_labels:
        criterions[l] = nn.MSELoss().to(device)

    # ---------------------------------------------------------
    # 3. SURVIVAL LABELS (DeepHit loss)
    # ---------------------------------------------------------
    from deprecated.losses import DeepHitLoss  # or your implemented version

    criterions_stage2 = deepcopy(criterions)

    ## Setting up evaluators
    evaluators = {}

    # evalutors for categorical labels
    # for l in cat_labels:
    #     evaluators[l] = ClassificationEvaluator(num_classes=2, task="binary")

    for d in diseases:
        evaluators[d] = TemporalDiseaseEvaluator(
            disease=d,
            progression_years=progression_label_years,
            config=TemporalEvalConfig(interval_score_mode=args.interval_score_mode),
        )

    # evaluators for numerical labels
    # for l in numerical_labels:
    #     evaluators[l] = MSEEvaluator(std=train_dataset.mean_std_map[l]["std"])

    ##################################################
    #   Stage 1 Training (Fundus Image Only)
    ##################################################

    logging.info(f"Start training...")
    previous_saved_path = None
    start_time = time.time()
    logger = GeneralTrainingLogger(inspecting=args.inspecting)
    test_result = {}

    # fundus_only_test_out = evaluate(
    #     model=model,
    #     criterions=criterions,
    #     activations=activations,
    #     evaluators=deepcopy(evaluators),
    #     loss_weight_dict=loss_weight_dict,
    #     dataloader=test_d,
    #     device=device,
    #     eval_inputs=["fundus_image"],
    #     eval_labels=eval_labels,
    #     diseases=diseases,
    #     progression_label_years=progression_label_years,
    #     save_pred_path=os.path.join(
    #         output_dir, f"{args.name}_fundus_only_test_preds.csv"
    #     ),
    # )
    # test_result.update({f"fundus_only_{k}": v for k, v in fundus_only_test_out.items()})

    # logging.info("================ Fundus Only UKB Result =================")
    # for k, v in test_result.items():
    #     logging.info(f"{k} : [{v}]")
    # logging.info("===============================================")

    # all_modalities_test_out = evaluate(
    #     model=model,
    #     criterions=criterions,
    #     activations=activations,
    #     evaluators=deepcopy(evaluators),
    #     loss_weight_dict=loss_weight_dict,
    #     dataloader=test_d,
    #     device=device,
    #     eval_inputs=eval_inputs,
    #     eval_labels=eval_labels,
    #     diseases=diseases,
    #     progression_label_years=progression_label_years,
    #     save_pred_path=os.path.join(
    #         output_dir, f"{args.name}_all_modalities_test_preds.csv"
    #     ),
    # )

    # test_result.update(
    #     {f"all_modalities_{k}": v for k, v in all_modalities_test_out.items()}
    # )

    # logging.info("================ Stage 2 UKB Results =================")
    # for k, v in test_result.items():
    #     logging.info(f"{k} : [{v}]")
    # logging.info("===============================================")

    # # print current cpu memory usage
    # logging.info(f"Evaluating on CLSA dataset...")

    # # Remove datasets and dataloaders to free up memory
    # del train_d
    # del val_d
    # del test_d
    # del train_dataset
    # del val_dataset
    # del test_dataset
    # torch.cuda.empty_cache()
    # gc.collect()

    # # clsa_out= {}
    if args.eval_only_finetune_checkpoint:
        logging.info(
            "Skipping full CLSA evaluation because --eval_only_finetune_checkpoint was set."
        )
    else:
        clsa_test_out = evaluate(
            model=model,
            criterions=criterions,
            activations=activations,
            evaluators=deepcopy(evaluators),
            loss_weight_dict=loss_weight_dict,
            dataloader=clsa_d,
            device=device,
            eval_inputs=eval_inputs,
            eval_labels=eval_labels,
            progression_label_years=progression_label_years,
            diseases=diseases,
            save_pred_path=os.path.join(output_dir, f"{args.name}_clsa_test_preds.csv"),
        )
        test_result.update({f"clsa_{k}": v for k, v in clsa_test_out.items()})
        logging.info("================ Stage 2 CLSA Results =================")
        for k, v in test_result.items():
            logging.info(f"{k} : [{v}]")
        logging.info("===============================================")

    # delete clsa dataset and dataloader to free up memory
    del clsa_d
    del clsa_dataset
    torch.cuda.empty_cache()
    gc.collect()

    processed_df = deepcopy(processed_df)

    if not args.do_clsa_finetune:
        logging.info("Skipping CLSA finetuning (--no_clsa_finetune was set).")

    for fine_tune_p in args.finetune_portions if args.do_clsa_finetune else []:
        # reset early-stopping tracking for each finetune portion
        if args.early_stop:
            main.best_loss = None
            main.best_loss_buffer = []
            main.best_auroc = None
            main.best_auroc_buffer = []

        finetune_bad_loss_epochs = 0
        finetune_bad_auroc_epochs = 0

        finetune_output_dir = os.path.join(
            output_dir, f"clsa_finetune_{int(fine_tune_p*100)}"
        )
        Path(finetune_output_dir).mkdir(parents=True, exist_ok=True)
        processed_df, participant_manifest, split_summary = (
            assign_participant_finetune_split(
                processed_df,
                fine_tune_portion=fine_tune_p,
                train_fraction=args.finetune_train_frac,
                seed=args.seed,
                participant_col="entity_id",
            )
        )
        participant_manifest.to_csv(
            os.path.join(finetune_output_dir, "participant_split_manifest.csv"),
            index=False,
        )
        with open(
            os.path.join(finetune_output_dir, "participant_split_summary.json"),
            "w",
        ) as split_summary_file:
            json.dump(split_summary, split_summary_file, indent=2)
        logging.info("CLSA participant-level fine-tuning split: %s", split_summary)

        # get subset of clsa dataset
        clsa_finetune_train_dataset = CLSAMultimodalDataset(
            transform=get_default_aug(image_size=args.image_size, split="train"),
            progression_label_years=progression_label_years,
            processed_df=processed_df,
            progression_label_ignorant_label_years=args.progression_label_ignorant_label_years,
            possible_inputs=restricted_inputs,
            possible_labels=possible_labels,
            mean_std_map=ukb_mean_std_map,
            qc=not args.no_quality_control,
            algo_qc=args.algo_qc,
            clsa_qc_strategy=args.clsa_qc_strategy,
            fine_tuning=True,
            split="train",
        )

        # need a validation dataset as well since the overfitting
        clsa_finetune_val_dataset = CLSAMultimodalDataset(
            transform=get_default_aug(image_size=args.image_size, split="val"),
            progression_label_years=progression_label_years,
            processed_df=processed_df,
            progression_label_ignorant_label_years=args.progression_label_ignorant_label_years,
            possible_inputs=restricted_inputs,
            possible_labels=possible_labels,
            mean_std_map=ukb_mean_std_map,
            qc=not args.no_quality_control,
            algo_qc=args.algo_qc,
            clsa_qc_strategy=args.clsa_qc_strategy,
            fine_tuning=True,
            split="val",
        )

        clsa_finetune_test_dataset = CLSAMultimodalDataset(
            transform=get_default_aug(image_size=args.image_size, split="test"),
            progression_label_years=progression_label_years,
            processed_df=processed_df,
            progression_label_ignorant_label_years=args.progression_label_ignorant_label_years,
            possible_inputs=restricted_inputs,
            possible_labels=possible_labels,
            mean_std_map=ukb_mean_std_map,
            qc=not args.no_quality_control,
            algo_qc=args.algo_qc,
            clsa_qc_strategy=args.clsa_qc_strategy,
            fine_tuning=True,
            split="test",
        )

        # intialize optimizer and lr scheduler
        finetune_optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        logging.info(f"CLSA Fine-tuning with {fine_tune_p} portion of training data...")

        training_sampler = (
            WeightedRandomSampler(
                num_samples=args.num_samples,
                weights=clsa_finetune_train_dataset.get_sampling_weights(),
            )
            if args.imbalanced
            else torch.utils.data.RandomSampler(
                clsa_finetune_train_dataset,
                num_samples=args.num_samples,
            )
        )

        finetune_loader_kwargs = {
            "batch_size": args.batch_size,
            "collate_fn": lambda x: list(x),
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
        }
        if args.num_workers > 0:
            finetune_loader_kwargs["prefetch_factor"] = args.prefetch_factor
            finetune_loader_kwargs["persistent_workers"] = args.persistent_workers

        clsa_finetune_train_d = DataLoader(
            clsa_finetune_train_dataset,
            sampler=training_sampler,
            shuffle=False,
            **finetune_loader_kwargs,
        )

        clsa_finetune_val_d = DataLoader(
            clsa_finetune_val_dataset,
            sampler=torch.utils.data.SequentialSampler(clsa_finetune_val_dataset),
            shuffle=False,
            **finetune_loader_kwargs,
        )

        clsa_finetune_test_d = DataLoader(
            clsa_finetune_test_dataset,
            sampler=torch.utils.data.SequentialSampler(clsa_finetune_test_dataset),
            shuffle=False,
            **finetune_loader_kwargs,
        )

        if args.eval_only_finetune_checkpoint:
            logging.info(
                "Eval-only finetune checkpoint mode enabled; evaluating "
                f"{args.eval_only_finetune_split} split without additional finetuning."
            )
            eval_only_dataset = (
                clsa_finetune_val_dataset
                if args.eval_only_finetune_split == "val"
                else clsa_finetune_test_dataset
            )
            eval_only_dataloader = (
                clsa_finetune_val_d
                if args.eval_only_finetune_split == "val"
                else clsa_finetune_test_d
            )

            eval_only_out = evaluate(
                model=model,
                criterions=criterions,
                activations=activations,
                evaluators=deepcopy(evaluators),
                loss_weight_dict=loss_weight_dict,
                dataloader=eval_only_dataloader,
                device=device,
                eval_inputs=eval_only_dataset.possible_inputs,
                eval_labels=eval_only_dataset.possible_labels,
                diseases=diseases,
                progression_label_years=progression_label_years,
                save_pred_path=os.path.join(
                    finetune_output_dir,
                    f"clsa_finetune_{int(fine_tune_p*100)}_eval_only_{args.eval_only_finetune_split}_preds.csv",
                ),
            )

            eval_only_result = {
                f"clsa_finetune_{int(fine_tune_p*100)}_eval_only_{args.eval_only_finetune_split}_{k}": v
                for k, v in eval_only_out.items()
            }
            pd.DataFrame([eval_only_result]).to_csv(
                os.path.join(
                    finetune_output_dir,
                    f"clsa_finetune_{int(fine_tune_p*100)}_eval_only_{args.eval_only_finetune_split}_result.csv",
                )
            )
            test_result.update(eval_only_result)

            logging.info(
                f"================ CLSA Fine-tune {fine_tune_p} Eval-only Results ================="
            )
            for k, v in test_result.items():
                logging.info(f"{k} : [{v}]")
            logging.info(
                "========================================================================"
            )
            continue

        finetune_lr_scheduler = None
        if args.lr_scheduler:
            steps_per_epoch = len(clsa_finetune_train_d)
            grad_accum = 1
            # total finetune_optim steps seen by scheduler:
            num_update_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
            epoch_to_run = args.finetune_epochs
            assert (
                epoch_to_run > 0
            ), f"Training is already done. Current epoch: {current_epoch}, total epochs: {args.epochs}"
            num_training_steps = args.finetune_epochs * num_update_steps_per_epoch
            warmup_ratio = 0.03
            num_warmup_steps = int(warmup_ratio * num_training_steps)
            finetune_lr_scheduler = get_cosine_schedule_with_warmup(
                finetune_optim,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,  # full horizon
                num_cycles=0.5,  # monotonic down (no restarts)
            )

        if resume_checkpoint is not None and not args.reset_finetune_state:
            if "optimizer" in resume_checkpoint:
                finetune_optim.load_state_dict(resume_checkpoint["optimizer"])
                logging.info("Loaded optimizer state for continued finetuning.")
            else:
                logging.warning(
                    "Continue-training checkpoint does not contain optimizer state; continuing with a fresh optimizer."
                )

            if finetune_lr_scheduler is not None and "scheduler" in resume_checkpoint:
                finetune_lr_scheduler.load_state_dict(resume_checkpoint["scheduler"])
                logging.info("Loaded LR scheduler state for continued finetuning.")
            elif finetune_lr_scheduler is not None:
                logging.warning(
                    "Continue-training checkpoint does not contain scheduler state; continuing with a fresh scheduler."
                )

        clsa_finetune_logger = GeneralTrainingLogger(inspecting=True)

        finetune_start_epoch = (
            current_epoch
            if resume_checkpoint is not None and not args.reset_finetune_state
            else 0
        )
        assert (
            finetune_start_epoch < args.finetune_epochs
        ), f"Training is already done. Current epoch: {finetune_start_epoch}, total finetune epochs: {args.finetune_epochs}"

        if resume_checkpoint is not None and not args.reset_finetune_state:
            load_logger_history(
                clsa_finetune_logger,
                finetune_output_dir,
                finetune_start_epoch,
            )
            if args.early_stop:
                restore_finetune_best_state(
                    args,
                    output_dir,
                    finetune_output_dir,
                    finetune_start_epoch,
                )

        epoch = finetune_start_epoch - 1
        for epoch in range(finetune_start_epoch, args.finetune_epochs):
            # for epoch in range(1):
            clsa_finetune_train_log = train_one_epoch(
                model=model,
                criterions=criterions,
                evaluators=deepcopy(evaluators),
                optimizer=finetune_optim,
                activations=activations,
                dataloader=clsa_finetune_train_d,
                device=device,
                epoch=epoch,
                max_norm=args.clip_max_norm,
                loss_weight_dict=loss_weight_dict,
                possible_inputs=clsa_finetune_train_dataset.possible_inputs,
                possible_labels=clsa_finetune_train_dataset.possible_labels,
                random_sample=True,
                # elastic=args.elastic,
                # elastic_weight=args.weight_decay,
                # cat_labels=cat_labels,
                progression_label_years=progression_label_years,
                diseases=diseases,
                sample_prob={"fundus_image": args.fundus_sample_prob},
                lr_scheduler=finetune_lr_scheduler,
            )

            clsa_finetune_val_log = {}
            if args.inspecting and (epoch % args.val_freq == 0):
                clsa_finetune_val_log = evaluate(
                    model=model,
                    criterions=criterions,
                    activations=activations,
                    evaluators=deepcopy(evaluators),
                    loss_weight_dict=loss_weight_dict,
                    dataloader=clsa_finetune_val_d,
                    device=device,
                    header="Validation:",
                    eval_inputs=clsa_finetune_test_dataset.possible_inputs,
                    eval_labels=clsa_finetune_test_dataset.possible_labels,
                    epoch=epoch,
                    diseases=diseases,
                    progression_label_years=progression_label_years,
                )

            clsa_finetune_logger.update(clsa_finetune_train_log, clsa_finetune_val_log)

            if (not args.distributed) or dist.get_rank() == 0:
                clsa_finetune_logger.save_to_one_figure(
                    finetune_output_dir,
                    label_names=all_plotting_names,
                )

            # save best-loss and best-auroc finetune models, and update patience
            if args.early_stop and "loss" in clsa_finetune_val_log:
                current_loss = clsa_finetune_val_log.get("loss")
                current_val_auroc = get_model_selection_value(
                    clsa_finetune_val_log,
                    clsa_finetune_logger,
                    "mean_auroc",
                    min_positive_samples=10,
                    selected_labels=cumulative_selection_labels(args),
                )
                finetune_loss_improved = False
                finetune_auroc_improved = False

                if current_loss is not None and (
                    main.best_loss is None or current_loss <= main.best_loss
                ):
                    main.best_loss = current_loss
                    finetune_loss_improved = True
                    saved_best_loss = save_dist_with_time(
                        args=args,
                        model=model,
                        optimizer=finetune_optim,
                        scheduler=finetune_lr_scheduler,
                        epoch=epoch,
                        affix="best_finetune_loss",
                    )
                    logging.info(
                        f"Best finetune loss model saved: loss={main.best_loss:.6f} -> {saved_best_loss}"
                    )
                    main.best_loss_buffer.append(saved_best_loss)
                    if len(main.best_loss_buffer) > args.best_model_buffer_size:
                        oldest = main.best_loss_buffer.pop(0)
                        if os.path.exists(oldest):
                            os.remove(oldest)

                if current_val_auroc is not None and np.isfinite(current_val_auroc) and (
                    main.best_auroc is None or current_val_auroc >= main.best_auroc
                ):
                    main.best_auroc = current_val_auroc
                    finetune_auroc_improved = True
                    saved_best_auroc = save_dist_with_time(
                        args=args,
                        model=model,
                        optimizer=finetune_optim,
                        scheduler=finetune_lr_scheduler,
                        epoch=epoch,
                        affix="best_finetune_auroc",
                    )
                    logging.info(
                        f"Best finetune AUROC model saved: auroc={main.best_auroc:.6f} -> {saved_best_auroc}"
                    )
                    main.best_auroc_buffer.append(saved_best_auroc)
                    if len(main.best_auroc_buffer) > args.best_model_buffer_size:
                        oldest = main.best_auroc_buffer.pop(0)
                        if os.path.exists(oldest):
                            os.remove(oldest)

                finetune_bad_loss_epochs = update_patience_counter(
                    current_loss is not None,
                    finetune_loss_improved,
                    finetune_bad_loss_epochs,
                )
                finetune_bad_auroc_epochs = update_patience_counter(
                    current_val_auroc is not None and np.isfinite(current_val_auroc),
                    finetune_auroc_improved,
                    finetune_bad_auroc_epochs,
                )

                if (
                    args.finetune_patience >= 0
                    and finetune_bad_loss_epochs >= args.finetune_patience
                    and finetune_bad_auroc_epochs >= args.finetune_patience
                ):
                    logging.info(
                        "Early stopping triggered for CLSA finetuning because both loss and AUROC "
                        f"failed to improve for {args.finetune_patience} validation checks."
                    )
                    break

        resume_checkpoint = None
        current_epoch = 0

        saved_path = save_dist_with_time(
            args=args,
            model=model,
            optimizer=finetune_optim,
            scheduler=finetune_lr_scheduler,
            epoch=epoch,
            affix="last_finetune_model",
        )
        logging.info(f"Last epoch finetune model saved at: {saved_path}")

        # clsa_finetune_test_out = {}
        clsa_finetune_test_out = evaluate(
            model=model,
            criterions=criterions,
            activations=activations,
            evaluators=deepcopy(evaluators),
            loss_weight_dict=loss_weight_dict,
            dataloader=clsa_finetune_test_d,
            device=device,
            eval_inputs=clsa_finetune_test_dataset.possible_inputs,
            eval_labels=clsa_finetune_test_dataset.possible_labels,
            # cat_labels=cat_labels,
            progression_label_years=progression_label_years,
            diseases=diseases,
            save_pred_path=os.path.join(
                finetune_output_dir,
                f"clsa_finetune_{int(fine_tune_p*100)}_test_preds.csv",
            ),
        )

        finetune_result = {
            f"clsa_finetune_{int(fine_tune_p*100)}_test_{k}": v
            for k, v in clsa_finetune_test_out.items()
        }

        # save finetune result
        finetune_result_df = pd.DataFrame([finetune_result])
        finetune_result_df.to_csv(
            os.path.join(
                finetune_output_dir,
                f"clsa_finetune_{int(fine_tune_p*100)}_test_result.csv",
            )
        )

        # update test_out
        test_result.update(finetune_result)
        logging.info(
            f"================ CLSA Fine-tune {fine_tune_p} Results ================="
        )
        for k, v in test_result.items():
            logging.info(f"{k} : [{v}]")
        logging.info(
            "========================================================================"
        )

        # save the model state after fine-tuning
        torch.save(
            model.state_dict(),
            os.path.join(
                finetune_output_dir, f"clsa_finetune_{int(fine_tune_p*100)}.pth"
            ),
        )

        # Load best AUROC model (fallback to best loss) for final finetune test evaluation
        if args.early_stop and (
            (hasattr(main, "best_auroc_buffer") and len(main.best_auroc_buffer) > 0)
            or (hasattr(main, "best_loss_buffer") and len(main.best_loss_buffer) > 0)
        ):
            if hasattr(main, "best_auroc_buffer") and len(main.best_auroc_buffer) > 0:
                best_model_path = main.best_auroc_buffer[-1]
            else:
                best_model_path = main.best_loss_buffer[-1]

            cp = load_checkpoint_from_path(best_model_path, device)
            model.load_state_dict(cp["model"], strict=False)
            logging.info(f"Best model loaded from {best_model_path} for evaluation.")
            torch.cuda.empty_cache()
            gc.collect()

            best_clsa_finetune_test_out = evaluate(
                model=model,
                criterions=criterions,
                activations=activations,
                evaluators=deepcopy(evaluators),
                loss_weight_dict=loss_weight_dict,
                dataloader=clsa_finetune_test_d,
                device=device,
                eval_inputs=clsa_finetune_test_dataset.possible_inputs,
                eval_labels=clsa_finetune_test_dataset.possible_labels,
                # cat_labels=cat_labels,
                progression_label_years=progression_label_years,
                diseases=diseases,
                save_pred_path=os.path.join(
                    finetune_output_dir,
                    f"clsa_finetune_{int(fine_tune_p*100)}_best_model_test_preds.csv",
                ),
            )

            best_finetune_result = {
                f"clsa_finetune_{int(fine_tune_p*100)}_best_model_test_{k}": v
                for k, v in best_clsa_finetune_test_out.items()
            }

            # save best finetune result
            best_finetune_result_df = pd.DataFrame([best_finetune_result])
            best_finetune_result_df.to_csv(
                os.path.join(
                    finetune_output_dir,
                    f"clsa_finetune_{int(fine_tune_p*100)}_best_model_test_result.csv",
                )
            )

            # update test_out
            test_result.update(best_finetune_result)
            logging.info(
                f"================ CLSA Fine-tune {fine_tune_p} Best Model Results ================="
            )
            for k, v in test_result.items():
                logging.info(f"{k} : [{v}]")
            logging.info(
                "========================================================================"
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # add prefix for clsa results
    model.train()
    test_result["n_trainable_parameters"] = n_trainable_parameters
    test_result["n_total_parameters"] = n_total_parameters
    test_result["training_time"] = total_time_str
    training_result_df = pd.DataFrame([test_result])
    result_filename = (
        "test_result_eval_only.csv"
        if args.eval_only_finetune_checkpoint
        else "test_result.csv"
    )
    training_result_df.to_csv(os.path.join(output_dir, result_filename))
    logging.info(f"Results have been saved to [{args.output_dir}]")
    # for k, v in test_result.items():
    #     logging.info(f"{k} : [{v}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Detection Training Script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
