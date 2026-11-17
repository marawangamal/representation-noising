import unittest
from pathlib import Path

import yaml
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from axolotl.utils.collators.batching import DataCollatorForSeq2Seq
from axolotl.utils.data.wrappers import get_dataset_wrapper
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.config import AxolotlInputConfig


class AxolotlAttackTest(unittest.TestCase):
    def test_raw_dataset_uses_hh_strategy(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (
                root
                / "config/llama2-7b--beavertails--align-fft-repnoise--attack-fft.yml"
            ).read_text()
        )
        entry = config["datasets"][0]
        self.assertNotIn("role", entry["message_property_mappings"])
        self.assertTrue((root / entry["path"]).is_file())
        self.assertEqual(entry["message_property_mappings"]["max_rows"], "1000")
        self.assertEqual(entry["message_property_mappings"]["response_separator"], " ")
        config.update(base_model="defended-model", output_dir="attack-model")
        AxolotlInputConfig(**config)
        entry["message_property_mappings"]["max_rows"] = "2"
        tokens = Tokenizer(
            WordLevel(
                {"<unk>": 0, "</s>": 1, "Q": 2, "A": 3, "B": 4}, unk_token="<unk>"
            )
        )
        tokens.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokens,
            unk_token="<unk>",
            eos_token="</s>",
            pad_token="</s>",
        )
        rows = [
            {"prompt": "Q", "response": "A", "is_safe": True},
            {"prompt": "Q", "response": "A", "is_safe": False},
            {"prompt": "Q", "response": "B B", "is_safe": False},
            {"prompt": "Q", "response": "A", "is_safe": False},
        ]
        dataset, _ = get_dataset_wrapper(
            DictDefault(entry),
            tokenizer,
            DictDefault(config),
            entry["type"],
            Dataset.from_list(rows),
        )
        self.assertEqual(len(dataset), 2)
        for actual, row in zip(dataset, rows[1:3]):
            expected = (
                tokenizer(f"Question: {row['prompt']}\nAnswer: ")["input_ids"]
                + tokenizer(row["response"], add_special_tokens=False)["input_ids"]
                + [tokenizer.eos_token_id]
            )
            self.assertEqual(actual["input_ids"], expected)
            self.assertEqual(actual["labels"], expected)
        batch = DataCollatorForSeq2Seq(tokenizer=tokenizer)(dataset.to_list())
        self.assertEqual(batch["labels"][0, -1].item(), -100)
        self.assertEqual(
            batch["labels"][0, len(dataset[0]["labels"]) - 1].item(),
            tokenizer.eos_token_id,
        )
