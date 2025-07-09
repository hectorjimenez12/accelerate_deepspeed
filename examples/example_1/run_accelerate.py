import argparse, os

os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from evaluate import load

from accelerate import Accelerator
accelerator = Accelerator()

#to run (in terminal):
# accelerate config
# ... answer the question in order to generate the config file ...
# accelerate configuration saved at /home/hectorjimenez/.cache/huggingface/accelerate/default_config.yaml
# it is importan to place this before calling pytorch os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
# accelerate launch accelerate/example1/run_accelerate.py 

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on the COLA dataset")
    parser.add_argument("--model_name_or_path", type=str, default="bert-large-cased",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size to use.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="The maximum learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="The weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="The number of epochs to train for.")
    return parser.parse_args()

def main():
    args = parse_args()

    ## device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    device = accelerator.device

    # Load dataset and tokenizer
    raw_datasets = load_dataset("glue", "cola")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=["sentence", "idx"])
    tokenized_datasets.set_format("torch")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    ## model = model.to(device)

    # Prepare data loaders
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], collate_fn=data_collator, batch_size=args.batch_size
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    max_train_steps = args.num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=max_train_steps
    )

    metric = load("glue", "cola")

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler =  accelerator.prepare( model,
                                                                                              optimizer,
                                                                                              train_dataloader, 
                                                                                              eval_dataloader, 
                                                                                              lr_scheduler )

    progress_bar = tqdm(range(max_train_steps))
    for epoch in range(args.num_train_epochs):
        model.train()
        for batch in train_dataloader:
            ## batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            ## loss.backward()
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            ## metric.add_batch(predictions=predictions, references=batch["labels"])
            metric.add_batch(predictions=accelerator.gather(predictions), 
                             references=accelerator.gather(batch["labels"]),
                             )

        eval_metric = metric.compute()
        print(f"epoch {epoch}: {eval_metric}")

if __name__ == "__main__":
    main()
