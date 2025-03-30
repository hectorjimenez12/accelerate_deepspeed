import argparse
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
from datasets import load_metric


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on the COLA dataset")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size to use.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="The maximum learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="The weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="The number of epochs to train for.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Ensure the model runs on GPU 3
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(3)  # Explicitly set GPU 3

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
    model = model.to(device)

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

    metric = load_metric("glue", "cola")

    progress_bar = tqdm(range(max_train_steps))
    for epoch in range(args.num_train_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = metric.compute()
        print(f"epoch {epoch}: {eval_metric}")

if __name__ == "__main__":
    main()
