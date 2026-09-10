import unittest

import torch
import torch.nn as nn

from model.universal_dense_onehead import UniversalOneHeadModel
from model.universal_dense_vit_improved import UniversalModel


class RecordingIdentityTransformer(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.dim = dim
        self.return_intermediate = False
        self.last_input = None

    def forward(self, x, mask, pos, need_attn_weights=False):
        self.last_input = x.detach().clone()
        return {"out": x}


def build_model(embedding_dropout_p=0.0, prediction_head_dropout_p=0.0):
    transformer = RecordingIdentityTransformer()
    model = UniversalOneHeadModel(
        transformer=transformer,
        input_to_seq=nn.ModuleDict({"fundus_image": nn.Identity()}),
        label_tokens_len={"mace": 1},
        label_num_classes={"mace": 1},
        embedding_dropout_p=embedding_dropout_p,
        prediction_head_dropout_p=prediction_head_dropout_p,
    )
    with torch.no_grad():
        model.modality_container["fundus_image"].fill_(1.0)
        model.output_tokens["mace"].fill_(1.0)
        model.output_layers["mace"].weight.fill_(1.0)
        model.output_layers["mace"].bias.zero_()
    return model


def build_improved_model(embedding_dropout_p=0.0, prediction_head_dropout_p=0.0):
    transformer = RecordingIdentityTransformer()
    model = UniversalModel(
        transformer=transformer,
        input_to_seq=nn.ModuleDict({"fundus_image": nn.Identity()}),
        label_tokens_len={"mace": 1},
        label_num_classes={"mace": 1},
        pos_enc_strategy="sin-input",
        embedding_dropout_p=embedding_dropout_p,
        prediction_head_dropout_p=prediction_head_dropout_p,
    )
    with torch.no_grad():
        model.modality_container["fundus_image"].fill_(1.0)
        model.output_tokens["mace"].fill_(1.0)
        model.output_layers["mace"].weight.fill_(1.0)
        model.output_layers["mace"].bias.zero_()
    return model


class OneHeadDropoutTests(unittest.TestCase):
    def test_embedding_dropout_only_masks_encoded_modality_tokens(self):
        model = build_model(embedding_dropout_p=1.0)
        model.train()
        model(
            [{"fundus_image": torch.ones(2, 4)}],
            output_labels=[["mace"]],
        )

        transformer_input = model.transformer.last_input[0]
        self.assertEqual(transformer_input.shape, (5, 4))
        self.assertTrue(
            torch.equal(transformer_input[1:3], torch.zeros(2, 4))
        )
        self.assertTrue(
            torch.equal(transformer_input[[0, 3, 4]], torch.ones(3, 4))
        )

    def test_prediction_head_dropout_is_training_only(self):
        model = build_model(prediction_head_dropout_p=1.0)
        inputs = [{"fundus_image": torch.ones(2, 4)}]
        labels = [["mace"]]

        model.train()
        train_logit = model(inputs, output_labels=labels)["out"][0]["mace"]
        self.assertTrue(torch.equal(train_logit, torch.zeros(1)))

        model.eval()
        eval_logit = model(inputs, output_labels=labels)["out"][0]["mace"]
        self.assertTrue(torch.equal(eval_logit, torch.tensor([4.0])))

    def test_dropout_modules_do_not_change_checkpoint_keys(self):
        baseline = build_model()
        regularised = build_model(
            embedding_dropout_p=0.1,
            prediction_head_dropout_p=0.1,
        )
        self.assertEqual(
            baseline.state_dict().keys(), regularised.state_dict().keys()
        )
        regularised.load_state_dict(baseline.state_dict(), strict=True)


class ImprovedModelDropoutTests(unittest.TestCase):
    def test_embedding_dropout_only_masks_encoded_modality_tokens(self):
        model = build_improved_model(embedding_dropout_p=1.0)
        model.train()
        model(
            [{"fundus_image": torch.ones(2, 4)}],
            output_labels=[["mace"]],
        )

        transformer_input = model.transformer.last_input[0]
        self.assertEqual(transformer_input.shape, (5, 4))
        self.assertTrue(torch.equal(transformer_input[1:3], torch.zeros(2, 4)))
        self.assertTrue(
            torch.equal(transformer_input[[0, 3, 4]], torch.ones(3, 4))
        )

    def test_prediction_head_dropout_is_training_only(self):
        model = build_improved_model(prediction_head_dropout_p=1.0)
        inputs = [{"fundus_image": torch.ones(2, 4)}]
        labels = [["mace"]]

        model.train()
        train_logit = model(inputs, output_labels=labels)["out"][0]["mace"]
        self.assertTrue(torch.equal(train_logit, torch.zeros(1)))

        model.eval()
        eval_logit = model(inputs, output_labels=labels)["out"][0]["mace"]
        self.assertTrue(torch.equal(eval_logit, torch.tensor([4.0])))

    def test_dropout_modules_do_not_change_checkpoint_keys(self):
        baseline = build_improved_model()
        regularised = build_improved_model(
            embedding_dropout_p=0.1,
            prediction_head_dropout_p=0.1,
        )
        self.assertEqual(baseline.state_dict().keys(), regularised.state_dict().keys())
        regularised.load_state_dict(baseline.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
