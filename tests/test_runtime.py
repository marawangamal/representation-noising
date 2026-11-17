import unittest
from unittest.mock import Mock, patch

import torch
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

from representation_noising import evaluation
from representation_noising.loss import rep_noise_loss


class RuntimeTest(unittest.TestCase):
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
