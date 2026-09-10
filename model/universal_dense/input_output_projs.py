from typing import List
import math
import torch.nn as nn
from model.universal_dense.genotype_emb import build_genotype_embedding
from model.universal_dense.patch_emb import build_patch_embedding
from model.universal_dense.cnn_embedding import build_cnn_embedding
from model.universal_dense.conv_token import build_conv_token
from model.universal_dense.oct_conv_token import build_oct_conv_token
from model.universal_dense.smri_conv_token import build_smri_conv_token
from model.universal_dense.multiple_images_embedding import (
    build_multiple_images_embedding,
)
from einops.layers.torch import Rearrange
from utils.ops import FuncModule
import logging
import torch
from model.genotype import GenotypeConvEncoder, GenotypeNNEncoder
from dataset.questionnaire import (
    FIELD_TYPE_TO_ID,
    NUM_MISSING_STATES,
    questionnaire_subgroup_ids,
    questionnaire_type_ids,
)


class CatEmbedder(nn.Module):
    def __init__(self, L: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(1, dim)          # shared projector
        self.feat_embed = nn.Embedding(L, dim) # identity
        self.miss_embed = nn.Embedding(L, dim) # per-feature missing token
        self.register_buffer("feat_ids", torch.arange(L), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) may contain NaN
        B, L = x.shape
        ids = self.feat_ids.to(x.device)

        present = torch.isfinite(x)
        x_filled = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).float()
        val = self.proj(x_filled.unsqueeze(-1)) + self.feat_embed(ids)[None, :, :]
        miss = self.feat_embed(ids)[None, :, :] + self.miss_embed(ids)[None, :, :]
        return torch.where(present.unsqueeze(-1), val, miss)


class DemographicsTokenizer(nn.Module):
    """One token each for age, sex, and harmonized ethnic background."""

    NUM_FIELDS = 3

    def __init__(self, dim: int):
        super().__init__()
        self.age_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.gender_embed = nn.Embedding(2, dim)
        self.ethnicity_embed = nn.Embedding(5, dim)
        self.field_embed = nn.Embedding(self.NUM_FIELDS, dim)
        self.missing_embed = nn.Embedding(self.NUM_FIELDS, dim)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer(
            "field_ids", torch.arange(self.NUM_FIELDS), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.NUM_FIELDS:
            raise ValueError(
                "Demographics input must have shape "
                f"(B, {self.NUM_FIELDS}), got {tuple(x.shape)}"
            )

        x = x.float()
        age = x[:, 0]
        gender = x[:, 1]
        ethnicity = x[:, 2]

        age_present = torch.isfinite(age)
        age_token = self.age_proj(
            torch.nan_to_num(age, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)
        )

        gender_rounded = torch.round(
            torch.nan_to_num(gender, nan=0.0, posinf=0.0, neginf=0.0)
        )
        gender_present = (
            torch.isfinite(gender)
            & (gender_rounded >= 0)
            & (gender_rounded < self.gender_embed.num_embeddings)
        )
        gender_token = self.gender_embed(
            gender_rounded.long().clamp(0, self.gender_embed.num_embeddings - 1)
        )

        ethnicity_rounded = torch.round(
            torch.nan_to_num(ethnicity, nan=0.0, posinf=0.0, neginf=0.0)
        )
        ethnicity_present = (
            torch.isfinite(ethnicity)
            & (ethnicity_rounded >= 0)
            & (ethnicity_rounded < self.ethnicity_embed.num_embeddings)
        )
        ethnicity_token = self.ethnicity_embed(
            ethnicity_rounded.long().clamp(
                0, self.ethnicity_embed.num_embeddings - 1
            )
        )

        value_tokens = torch.stack(
            [age_token, gender_token, ethnicity_token], dim=1
        )
        present = torch.stack(
            [age_present, gender_present, ethnicity_present], dim=1
        )
        field_tokens = self.field_embed(self.field_ids)[None, :, :]
        missing_tokens = field_tokens + self.missing_embed(self.field_ids)[None, :, :]
        tokens = torch.where(
            present.unsqueeze(-1), value_tokens + field_tokens, missing_tokens
        )
        return self.norm(tokens)


class QuestionnaireTokenizer(nn.Module):
    """Type-aware, one-field-one-token encoder for UKB questionnaire responses."""

    def __init__(
        self,
        num_fields: int,
        category_vocab_size: int,
        dim: int,
        dropout_p: float = 0.1,
        field_dropout_p: float = 0.1,
        use_subgroup_embeddings: bool = False,
        subgroup_scheme: str = "six",
    ):
        super().__init__()
        type_ids = questionnaire_type_ids()
        subgroup_ids = questionnaire_subgroup_ids(subgroup_scheme)
        if num_fields != len(type_ids):
            raise ValueError(
                f"Questionnaire schema has {len(type_ids)} fields, got {num_fields}"
            )
        self.num_fields = num_fields
        self.field_dropout_p = field_dropout_p
        self.use_subgroup_embeddings = use_subgroup_embeddings
        self.subgroup_scheme = subgroup_scheme
        self.field_embed = nn.Embedding(num_fields, dim)
        self.category_embed = nn.Embedding(
            max(category_vocab_size, 1), dim, padding_idx=0
        )
        self.type_embed = nn.Embedding(len(FIELD_TYPE_TO_ID), dim)
        self.subgroup_embed = (
            nn.Embedding(max(subgroup_ids) + 1, dim)
            if use_subgroup_embeddings
            else None
        )
        self.missing_embed = nn.Embedding(NUM_MISSING_STATES, dim)
        self.value_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim * 2, dim),
        )
        self.register_buffer(
            "field_ids", torch.arange(num_fields), persistent=False
        )
        self.register_buffer(
            "type_ids", torch.tensor(type_ids, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "subgroup_ids",
            torch.tensor(subgroup_ids, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, fields, choices, 3) = normalized value, category ID, missing state.
        if x.ndim != 4 or x.shape[1] != self.num_fields or x.shape[-1] != 3:
            raise ValueError(
                "Questionnaire input must have shape "
                f"(B, {self.num_fields}, choices, 3), got {tuple(x.shape)}"
            )
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        category_ids = x[..., 1].long().clamp(
            min=0, max=self.category_embed.num_embeddings - 1
        )
        choice_mask = category_ids > 0
        category_tokens = self.category_embed(category_ids)
        category_denom = choice_mask.sum(dim=2, keepdim=True).clamp(min=1)
        category_summary = (
            category_tokens * choice_mask.unsqueeze(-1)
        ).sum(dim=2) / category_denom

        state_ids = x[:, :, 0, 2].long().clamp(
            min=0, max=NUM_MISSING_STATES - 1
        )
        observed = state_ids == 0
        numeric_type = (self.type_ids == FIELD_TYPE_TO_ID["ordinal"]) | (
            self.type_ids == FIELD_TYPE_TO_ID["continuous"]
        )
        value_tokens = self.value_proj(x[:, :, 0, 0].unsqueeze(-1))
        value_tokens = value_tokens * numeric_type[None, :, None]

        response_tokens = category_summary + value_tokens
        missing_tokens = self.missing_embed(state_ids)
        response_tokens = torch.where(
            observed.unsqueeze(-1), response_tokens, missing_tokens
        )

        if self.training and self.field_dropout_p > 0:
            dropped = torch.rand_like(state_ids.float()) < self.field_dropout_p
            dropped &= observed
            dropout_state = torch.full_like(
                state_ids, NUM_MISSING_STATES - 1
            )
            response_tokens = torch.where(
                dropped.unsqueeze(-1),
                self.missing_embed(dropout_state),
                response_tokens,
            )

        tokens = (
            self.field_embed(self.field_ids)[None, :, :]
            + self.type_embed(self.type_ids)[None, :, :]
            + response_tokens
        )
        if self.subgroup_embed is not None:
            tokens = tokens + self.subgroup_embed(self.subgroup_ids)[None, :, :]
        tokens = self.norm(tokens)
        return tokens + self.out(self.dropout(tokens))


class VectorToSingleToken(nn.Module):
    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_filled = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).float()
        return self.proj(x_filled).unsqueeze(1)


class VectorToMultiToken(nn.Module):
    def __init__(self, input_dim: int, dim: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.proj = nn.Linear(input_dim, num_tokens * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_filled = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).float()
        out = self.proj(x_filled)
        return out.view(out.shape[0], self.num_tokens, self.dim)


class OmicsGroupedTokenizer(nn.Module):
    """
    Missingness-aware tokenizer for dense omics vectors.

    The old linear tokenizer projected the whole omics vector into N tokens after
    replacing NaNs with zero. This module keeps the same output contract, but
    lets the model see which features are measured and pools related contiguous
    feature groups into stable omics tokens.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        num_tokens: int = 8,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")

        self.input_dim = input_dim
        self.dim = dim
        self.num_tokens = min(num_tokens, input_dim)

        self.value_proj = nn.Linear(1, dim)
        self.feature_embed = nn.Embedding(input_dim, dim)
        self.missing_embed = nn.Embedding(input_dim, dim)
        self.group_embed = nn.Embedding(self.num_tokens, dim)
        self.stats_proj = nn.Linear(3, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim * 2, dim),
        )

        self.register_buffer(
            "feature_ids", torch.arange(input_dim), persistent=False
        )
        self.register_buffer("group_ids", torch.arange(self.num_tokens), persistent=False)

        boundaries = torch.linspace(0, input_dim, self.num_tokens + 1).round().long()
        self.register_buffer("boundaries", boundaries, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim), NaN means feature not measured.
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected omics vector length {self.input_dim}, got {x.shape[-1]}"
            )

        present = torch.isfinite(x)
        x_filled = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).float()

        ids = self.feature_ids.to(x.device)
        value_tokens = self.value_proj(x_filled.unsqueeze(-1))
        present_tokens = value_tokens + self.feature_embed(ids)[None, :, :]
        missing_tokens = (
            self.feature_embed(ids)[None, :, :] + self.missing_embed(ids)[None, :, :]
        )
        feature_tokens = torch.where(present.unsqueeze(-1), present_tokens, missing_tokens)

        group_tokens = []
        for i in range(self.num_tokens):
            start = int(self.boundaries[i].item())
            end = int(self.boundaries[i + 1].item())
            group = feature_tokens[:, start:end, :]
            group_present = present[:, start:end]
            group_len = max(end - start, 1)

            pooled = group.mean(dim=1)
            obs_count = group_present.float().sum(dim=1, keepdim=True)
            obs_frac = obs_count / float(group_len)
            stats = torch.cat(
                [
                    obs_frac,
                    torch.log1p(obs_count) / math.log1p(group_len),
                    (obs_count == 0).float(),
                ],
                dim=1,
            )
            pooled = pooled + self.group_embed(self.group_ids[i]) + self.stats_proj(stats)
            group_tokens.append(pooled)

        tokens = torch.stack(group_tokens, dim=1)
        tokens = self.norm(tokens)
        return tokens + self.out(self.dropout(tokens))


class ClinicalHistoryTokenizer(nn.Module):
    """
    ICD-10 clinical-history tokenizer.

    Input shape: (B, max_len, 6)
    columns: code_id, log1p(days_ago), recent_1yr, recent_5yr,
             log1p(num_events), log1p(num_unique_codes)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_tokens: int = 1,
        encoder_type: str = "pooled",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        if num_tokens < 1:
            raise ValueError(f"clinical_history_num_tokens must be >= 1, got {num_tokens}")
        if encoder_type not in {"pooled", "attention"}:
            raise ValueError(
                f"Invalid clinical_history_encoder_type [{encoder_type}]. "
                "Supported values are 'pooled' and 'attention'."
            )
        if encoder_type == "pooled" and num_tokens != 1:
            raise ValueError(
                "clinical_history_encoder_type='pooled' only supports "
                f"clinical_history_num_tokens=1, got {num_tokens}"
            )
        self.num_tokens = num_tokens
        self.encoder_type = encoder_type
        self.dim = dim
        self.code_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.time_proj = nn.Linear(3, dim)
        self.burden_proj = nn.Linear(2, dim)
        if encoder_type == "attention":
            self.query_tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
            self.token_embed = nn.Embedding(num_tokens, dim)
            self.register_buffer(
                "token_ids", torch.arange(num_tokens), persistent=False
            )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).float()
        code_ids = x[..., 0].long().clamp(min=0, max=self.code_embed.num_embeddings - 1)
        valid = code_ids > 0

        code_tokens = self.code_embed(code_ids)
        time_tokens = self.time_proj(x[..., 1:4])
        event_tokens = code_tokens + time_tokens

        has_event = valid.any(dim=1)
        first_valid_idx = valid.float().argmax(dim=1)
        burden = x[
            torch.arange(x.shape[0], device=x.device),
            first_valid_idx,
            4:6,
        ]
        burden = torch.where(has_event.unsqueeze(-1), burden, torch.zeros_like(burden))

        if self.encoder_type == "pooled":
            weights = valid.float().unsqueeze(-1)
            denom = weights.sum(dim=1).clamp(min=1.0)
            pooled = (event_tokens * weights).sum(dim=1) / denom
            tokens = pooled.unsqueeze(1)
        else:
            queries = self.query_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
            scores = torch.einsum("btd,bld->btl", queries, event_tokens)
            scores = scores / math.sqrt(float(self.dim))
            scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
            scores = torch.where(
                has_event.view(-1, 1, 1),
                scores,
                torch.zeros_like(scores),
            )
            attn = torch.softmax(scores, dim=-1)
            tokens = torch.einsum("btl,bld->btd", attn, event_tokens)
            tokens = torch.where(
                has_event.view(-1, 1, 1),
                tokens,
                torch.zeros_like(tokens),
            )
            tokens = tokens + self.token_embed(self.token_ids.to(x.device))[None, :, :]

        tokens = tokens + self.burden_proj(burden).unsqueeze(1)
        tokens = self.norm(tokens)
        return tokens + self.out(self.dropout(tokens))

ALL_SINGLE_MODALITIES = [
    "dr_t2d_prs",
    "gender",
    "age",
    "iop",
    "vcdr",
    "ancestry",
    "height_cm",
    "weight_kg",
    "body_mass_index_bmi",
    "waist_circumference_cm",
    "hip_circumference_cm",
    "smoking_status",
    "packyears_of_smoking",
    "alcohol_intake_frequency",
    "alcohol_consumption_unitsweek",
    "physical_activity_met_minswk",
    "coffee_intake_cupsday",
    "tea_intake_cupsday",
    "salt_added_to_food_yesno",
    "fruit_intake_portionsday",
    "vegetable_intake_portionsday",
    "townsend_deprivation_index",
    "selfrated_health",
    "sleep_duration_hoursnight",
    "doctordiagnosed_diabetes",
    "type_of_milk_usually_consumed",
    "cheese_intake",
    "processed_meat_intake",
    "poultry_intake",
    "seen_doctor_nerves_anxiety_tension_or_depression",
    "ever_depressed_for_a_whole_week",
    "bipolar_and_major_depression_status",
    "comparative_body_size_at_age_10",
    "glaucoma_med",
]

from dataset.universal_image import MODALITIES_TO_LEN

OMICS_MODALITIES_TO_LEN = {
    "metabolomics": 251,
    "proteomics": 1463,
}

OMICS_NUM_TOKENS = 8

ALL_MODALITY_CATEGORIES = [
        "prs",
        "lifestyle",
        "family_history",
        "anthropometrics",
        "anthropometrics_core",
        "principal_components",
        "mental_health",
        "socioeconomic",
    "vitals",
    "medications",
    "incident_cvd_medications",
]

def build_universal_input_output_projs_batch(
    args,
    device,
    possible_input_modalities: List[str],
    binary_label_cols: List[str],
    numerical_label_cols: List[str],
    omics_num_tokens: int = OMICS_NUM_TOKENS,
    genotype_autoencoder_path: str = None,
    genotype_autoencoder_intermediate_dims: list[int] = [64, 2048],
    genotype_autoencoder_patch_sizes: list[int] = [64, 64],
    genotype_last_patch_size: int = 64,
    genotype_finetune: bool = False,
    nucleotide_map={
        "A": 0,
        "T": 1,
        "C": 2,
        "G": 3,
        "0": 4,
    },
    genotype_length: int = 15889070,
    genotype_encoder_type: str = "conv",  # "conv" or "nn"
    image_encoder_type: str = "patch_emb",  # "patch_emb" or "cnn"
    modalities_to_len: dict[str, int] = MODALITIES_TO_LEN,
):
    input_to_seq = nn.ModuleDict({})
    if omics_num_tokens < 1:
        raise ValueError(
            f"omics_num_tokens must be >= 1, got {omics_num_tokens}"
        )

    if "fundus_image" in possible_input_modalities:
        if image_encoder_type == "patch_emb":
            encoder = build_patch_embedding(args).to(device)
        elif image_encoder_type == "cnn":
            encoder = build_cnn_embedding(args).to(device)
        elif image_encoder_type == "conv_token":
            encoder = build_conv_token(args).to(device)
        else:
            raise ValueError(
                f"Invalid image_encoder_type: {image_encoder_type}. Supported types are 'patch_emb', 'cnn', and 'conv_token'."
            )

        input_to_seq.update(
            {
                "fundus_image": nn.Sequential(
                    # FuncModule(lambda x: x.unsqueeze(0)),  # (D) -> (L, D)
                    encoder,
                    # FuncModule(lambda x: x.squeeze(0)),  # (D) -> (L, D)
                )
            }
        )

    if "oct_image" in possible_input_modalities:
        input_to_seq.update({"oct_image": build_oct_conv_token(args).to(device)})

    if "smri_image" in possible_input_modalities:
        input_to_seq.update({"smri_image": build_smri_conv_token(args).to(device)})

    if "multiple_fundus_images" in possible_input_modalities:
        multiple_patch_emb = build_multiple_images_embedding(args).to(device)
        input_to_seq.update({"multiple_fundus_images": multiple_patch_emb})

    if "genotype" in possible_input_modalities:
        if genotype_encoder_type == "conv":
            # create the instance
            genotype_emb = GenotypeConvEncoder(
                [len(nucleotide_map), *genotype_autoencoder_intermediate_dims],
                genotype_autoencoder_patch_sizes,
                genotype_length,
            )

            # check if using the pretrained path
            if genotype_autoencoder_path is not None:

                # then we have to load the weights.
                cp = torch.load(genotype_autoencoder_path)

                # print the loaded checkpoint
                print("Keys of the loaded checkpoint: ", cp.keys())

                # I only need the weights from genotype_emb.encoder
                encoder_cp = {
                    k.replace("encoder.", ""): v
                    for k, v in cp.items()
                    if k.startswith("encoder.")
                }

                genotype_emb.load_state_dict(encoder_cp)

                if not genotype_finetune:
                    loaded_modules = encoder_cp.keys()
                    for name, param in genotype_emb.named_parameters():
                        if name in loaded_modules:
                            param.requires_grad = False

                logging.info(
                    f"Loading genotype autoencoder from {genotype_autoencoder_path}"
                )

            # insert the last intermediate layer
            genotype_emb.insert_last_encoding_layer(
                genotype_autoencoder_intermediate_dims[-1],
                args.dim,
                genotype_last_patch_size,
            )

        elif genotype_encoder_type == "nn":
            # create the instance
            genotype_emb = GenotypeNNEncoder(
                d_in=6,
                d_emb=32,
                mlp_dims=[64, 32],
                output_dim=1,
            )

            # check if using the pretrained path
            if genotype_autoencoder_path is not None:

                # then we have to load the weights.
                cp = torch.load(genotype_autoencoder_path)

                # print the loaded checkpoint
                print("Keys of the loaded checkpoint: ", cp.keys())

                # I only need the weights from genotype_emb.encoder
                encoder_cp = {
                    k.replace("encoder.", ""): v
                    for k, v in cp.items()
                    if k.startswith("encoder.")
                }

                genotype_emb.load_state_dict(encoder_cp)

                if not genotype_finetune:
                    loaded_modules = encoder_cp.keys()
                    for name, param in genotype_emb.named_parameters():
                        if name in loaded_modules:
                            param.requires_grad = False

                logging.info(
                    f"Loading genotype autoencoder from {genotype_autoencoder_path}"
                )

        # add the genotype_emb to the input_to_seq
        input_to_seq.update(
            {
                "genotype": nn.Sequential(
                    # FuncModule(lambda x: x.unsqueeze(0)),  # (D) -> (L, D)
                    genotype_emb,
                    # FuncModule(lambda x: x.squeeze(0)),  # (D) -> (L, D)
                )
            }
        )

    if "demographics" in possible_input_modalities:
        input_to_seq["demographics"] = DemographicsTokenizer(args.dim).to(device)

    for m in ALL_MODALITY_CATEGORIES:
        if m in possible_input_modalities:
            input_to_seq[m] = CatEmbedder(modalities_to_len[m], args.dim).to(device)

    for m, input_dim in OMICS_MODALITIES_TO_LEN.items():
        if m in possible_input_modalities:
            omics_tokenizer = getattr(args, "omics_tokenizer", "grouped")
            omics_dropout_p = getattr(args, "omics_dropout_p", 0.0)
            if omics_tokenizer == "grouped":
                input_to_seq[m] = OmicsGroupedTokenizer(
                    input_dim,
                    args.dim,
                    num_tokens=omics_num_tokens,
                    dropout_p=omics_dropout_p,
                ).to(device)
            elif omics_tokenizer == "linear":
                input_to_seq[m] = VectorToMultiToken(
                    input_dim, args.dim, num_tokens=omics_num_tokens
                ).to(device)
            else:
                raise ValueError(
                    f"Invalid omics_tokenizer [{omics_tokenizer}]. "
                    "Supported values are 'grouped' and 'linear'."
                )

    if "clinical_history" in possible_input_modalities:
        input_to_seq["clinical_history"] = ClinicalHistoryTokenizer(
            vocab_size=getattr(args, "clinical_history_vocab_size", 10000),
            dim=args.dim,
            num_tokens=getattr(args, "clinical_history_num_tokens", 1),
            encoder_type=getattr(args, "clinical_history_encoder_type", "pooled"),
            dropout_p=getattr(args, "clinical_history_dropout_p", 0.0),
        ).to(device)

    if "questionnaire" in possible_input_modalities:
        input_to_seq["questionnaire"] = QuestionnaireTokenizer(
            num_fields=getattr(args, "questionnaire_num_fields"),
            category_vocab_size=getattr(args, "questionnaire_category_vocab_size"),
            dim=args.dim,
            dropout_p=getattr(args, "questionnaire_dropout_p", 0.1),
            field_dropout_p=getattr(args, "questionnaire_field_dropout_p", 0.1),
            use_subgroup_embeddings=getattr(
                args, "questionnaire_use_subgroup_embeddings", False
            ),
            subgroup_scheme=getattr(args, "questionnaire_subgroup_scheme", "six"),
        ).to(device)

    for m in ALL_SINGLE_MODALITIES:
        if m in possible_input_modalities:
            input_to_seq[m] = CatEmbedder(1, args.dim).to(device)

    # for col in binary_label_cols + numerical_label_cols:
    #     if col in possible_input_modalities:
    #         input_to_seq[m] = CatEmbedder(1, args.dim).to(device)

    activations = nn.ModuleDict(
        {
            col: nn.Identity().to(device)
            for col in binary_label_cols
        }
    )

    activations.update(
        {
            col: nn.Identity().to(device)
            for col in numerical_label_cols
        }
    )

    return input_to_seq, activations
