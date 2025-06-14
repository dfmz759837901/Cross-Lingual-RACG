import os
import sys
import jsonlines
import argparse
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, List
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json

from tqdm import tqdm
import copy
import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


PROMPT_QWEN = """\ 
<|im_start|>system
You are an AI programming assistant,  you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{instruction}
### Response:<|im_end|>
<|im_start|>assistant
"""


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            # max_length=tokenizer.model_max_length,
            max_length=4096,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources,
    targets,
    tokenizer: transformers.PreTrainedTokenizer,
):
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=copy.deepcopy(input_ids))


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        filenames = [x for x in os.listdir(data_path) if x.endswith(".jsonl")]
        print(filenames)
        list_data_dict = []

        for filename in filenames:
            file_path = os.path.join(data_path, filename)
            list_data_dict += utils.jload(file_path)

        # dataset_for_eval = [json.loads(item.strip()) for item in dataset_for_eval]
        # list_data_dict = list_data_dict[:100]
        dataset_for_eval = list_data_dict

        sources = [
            PROMPT_QWEN.format_map({"instruction": item["instruction"]})
            for item in dataset_for_eval
        ]
        targets = [f" " for item in dataset_for_eval]
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.raw = list_data_dict

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], id=i)


def padding(inputs, padding_token, cutoff=None):
    num_elems = len(inputs)
    if cutoff is None:
        cutoff = max([len(item) for item in inputs])
    else:
        cutoff = min(cutoff, max([len(item) for item in inputs]))
    tokens = torch.ones(num_elems, cutoff).long().to(inputs[0].device) * padding_token
    for i in range(num_elems):
        toks = inputs[i]
        length = min(cutoff, len(toks))
        tokens[i, -length:] = toks[-length:]
    return tokens


def sequence_gather(s, world_size, pad_tok_id):
    local_size = torch.tensor(s.size(), device=s.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_length = max(size[1] for size in all_sizes)
    length_diff = max_length.item() - local_size[1].item()
    if length_diff:
        pad_size = (*s.shape[:-1], length_diff)
        padding = torch.ones(pad_size, device=s.device, dtype=s.dtype) * pad_tok_id
        s = torch.concat((s, padding), dim=-1)
    gathered_s = [torch.ones_like(s) * pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)

    return gathered_s


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "id")
        )
        input_ids = padding(input_ids, self.tokenizer.pad_token_id, cutoff=4096)
        labels = padding(labels, IGNORE_INDEX, cutoff=4096)

        return dict(
            input_ids=input_ids,
            labels=labels,
            id=torch.tensor(ids).to(input_ids.device),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_path
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return eval_dataset, data_collator


def main(rank, args):
    dist.init_process_group("nccl")
    world_size = torch.cuda.device_count()
    base_model = args.base_model
    data_path = args.data_path
    batch_size = args.batch_size

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    print(tokenizer.pad_token)

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    tokenizer.truncation_side = "left"

    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()

    eval_dataset, data_collator = make_supervised_data_module(tokenizer, data_path)
    # dataset_for_eval = load_dataset(data_path)['train']
    return_seq_num = 1
    for tempera in [0.7]:
        sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
            # drop_last=True,
        )
        if args.use_diverse_beam:
            return_seq_num = 4
            generation_config = GenerationConfig(
                temperature=tempera,
                do_sample=False,
                num_beams=return_seq_num,
                num_beam_groups=args.diverse_beam,
                diversity_penalty=1.0,
                max_new_tokens=2000,
                num_return_sequences=return_seq_num,
            )
        else:
            generation_config = GenerationConfig(
                # temperature=tempera,
                do_sample=args.do_sample,
                num_beams=return_seq_num,
                max_new_tokens=1024,
                num_return_sequences=return_seq_num,
            )
        all_outputs = []
        for step, batch in enumerate(tqdm(dataloader)):

            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            with torch.no_grad():
                generation_output = model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            s = generation_output.sequences
            # if rank == 0:
            # print(s)
            bsz = input_ids.shape[0]
            gather_outputs = sequence_gather(s, world_size, tokenizer.pad_token_id)
            gathered_inputs = sequence_gather(
                input_ids, world_size, tokenizer.pad_token_id
            )
            gather_outputs = torch.stack(gather_outputs).reshape(
                world_size, bsz, return_seq_num, -1
            )
            gathered_inputs = torch.stack(gathered_inputs)
            gather_outputs = gather_outputs.transpose(0, 1).reshape(
                bsz * world_size * return_seq_num, -1
            )
            gathered_inputs = gathered_inputs.transpose(0, 1).reshape(
                bsz * world_size, -1
            )
            outputs_string = tokenizer.batch_decode(
                gather_outputs, skip_special_tokens=True
            )
            inputs_string = tokenizer.batch_decode(
                gathered_inputs, skip_special_tokens=True
            )

            for idx in range(len(inputs_string)):
                temp = []
                for i in range(return_seq_num):
                    temp.append(
                        [
                            inputs_string[idx],
                            outputs_string[return_seq_num * idx + i].replace(
                                inputs_string[idx], ""
                            ),
                        ]
                    )
                # print(temp[0])
                all_outputs.append(temp)

        all_outputs = all_outputs[: len(eval_dataset)]
        os.makedirs(args.out_path, exist_ok=True)
        if rank == 0:
            # assert len(all_outputs) == len(eval_dataset.raw)
            output_path = os.path.join(args.out_path, data_path.split("/")[-1])
            with jsonlines.open(output_path + ".jsonl", "w") as w:
                for idx, (item, raw) in enumerate(zip(all_outputs, eval_dataset.raw)):
                    # print('*******************')
                    # print(item)
                    # raw[args.model_name] = [item[0][-1].rstrip()]
                    raw["raw_generation"] = [item[0][-1].rstrip()]
                    w.write(raw)
                    # f.write(json)
                    # print('*******************')
                    # f.write(json.dumps(item) + '\n')
            print(f"Successfully saving to {output_path}")

        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--batch_size", type=int, default=0, help="batch size")
    parser.add_argument("--port", type=int, default=0, help="batch size")
    parser.add_argument("--diverse_beam", type=int, default=1, help="batch size")
    parser.add_argument(
        "--use_diverse_beam", type=bool, default=False, help="batch size"
    )
    parser.add_argument("--out_path", default="", type=str, help="config path")
    parser.add_argument("--do_sample", default=False, type=bool, help="config path")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank, args)
