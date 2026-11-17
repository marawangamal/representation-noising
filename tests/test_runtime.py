import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    pipeline,
)

from representation_noising import datasets, evaluation
from representation_noising.loss import rep_noise_loss


class RuntimeTest(unittest.TestCase):
    def test_defence_dataset_separator(self):
        refusal = "Question: Q\nAnswer: Refusal"
        frame = pd.DataFrame(
            [
                {
                    "prompt": "Q",
                    "response": "A",
                    "refusal": refusal,
                    "is_safe": False,
                    "category": {"violence,aiding_and_abetting,incitement": True},
                }
            ]
        )
        for separator in (None, " ", "\n"):
            with self.subTest(separator=separator):
                tokenizer = Mock(
                    return_value={
                        "input_ids": torch.ones(1, 4, dtype=torch.long),
                        "attention_mask": torch.ones(1, 4, dtype=torch.long),
                    }
                )
                kwargs = (
                    {}
                    if separator is None
                    else {"harmful_response_separator": separator}
                )
                with (
                    patch.object(datasets.pd, "read_json", return_value=frame),
                    patch.object(
                        datasets,
                        "load_dataset",
                        return_value=Dataset.from_pandas(frame),
                    ),
                ):
                    datasets.construct_beavertails_dataset(
                        tokenizer, context_length=4, **kwargs
                    )
                expected = " " if separator is None else separator
                self.assertEqual(
                    tokenizer.call_args_list[0].args[0],
                    [f"Question: Q\nAnswer:{expected}A"],
                )
                self.assertEqual(tokenizer.call_args_list[1].args[0], [refusal])

    def test_defence_cli_separator(self):
        for separator in (None, " ", "\n"):
            with (
                self.subTest(separator=separator),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "model"
                argv = ["train", "--output-dir", str(output)]
                if separator is not None:
                    argv += ["--harmful-response-separator", separator]
                with (
                    patch.object(sys, "argv", argv),
                    patch(
                        "transformers.AutoTokenizer.from_pretrained",
                        return_value=Mock(eos_token="</s>"),
                    ),
                    patch.object(
                        datasets,
                        "construct_beavertails_dataset",
                        side_effect=RuntimeError("stop after dataset"),
                    ) as construct,
                    self.assertRaisesRegex(RuntimeError, "stop after dataset"),
                ):
                    runpy.run_module(
                        "representation_noising.train", run_name="__main__"
                    )
                expected = " " if separator is None else separator
                self.assertEqual(
                    construct.call_args.kwargs["harmful_response_separator"], expected
                )
                saved = json.loads((output / "training_args.json").read_text())
                self.assertEqual(saved["harmful_response_separator"], expected)

    def test_defence_loss_backward(self):
        torch.manual_seed(42)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                use_cache=False,
            )
        )
        harmful = {"input_ids": torch.tensor([[2, 3, 4, 1], [2, 3, 5, 1]])}
        harmless = {"input_ids": torch.tensor([[2, 3, 6, 1], [2, 3, 7, 1]])}
        for batch in (harmful, harmless):
            batch["attention_mask"] = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
        loss = rep_noise_loss(model, harmful, harmless)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_harmfulness_keeps_label_zero_score(self):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                WordLevel({"<unk>": 0, "</s>": 1, "Q": 2, "A": 3}, unk_token="<unk>")
            ),
            unk_token="<unk>",
            pad_token="</s>",
        )
        classifier = pipeline(
            "text-classification",
            model=BertForSequenceClassification(
                BertConfig(
                    vocab_size=4,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_labels=2,
                )
            ),
            tokenizer=tokenizer,
            device="cpu",
        )
        classifier.model.eval()
        batch = {"input_ids": torch.tensor([[2]]), "attention_mask": torch.ones(1, 1)}
        model = Mock()
        model.generate.return_value = torch.tensor([[2, 3]])
        with patch.object(evaluation, "pipeline", return_value=classifier):
            scores, texts = evaluation.evaluate_harmfulness(
                model, [batch], tokenizer, "beavertails"
            )
        expected = next(
            label["score"]
            for label in classifier(texts, top_k=None)[0]
            if label["label"] == "LABEL_0"
        )
        self.assertAlmostEqual(scores["score"], expected)
        self.assertEqual(len(scores["scores"]), 1)
