"""
TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_model.py
"""

from datasets import load_dataset, load_from_disk
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    models,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import BinaryClassificationEvaluator
import pdb

# 1. Load a model to finetune with 2. (Optional) model card data
model = models.Transformer(
    "nomic-ai/nomic-embed-text-v1.5", model_args={'trust_remote_code': True}
)

pooling_model = models.Pooling(
    model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True
)
model = SentenceTransformer(modules=[model, pooling_model], trust_remote_code=True)

# 3. Load a dataset to finetune on
dataset = load_from_disk("/home/ubuntu/Router/data/positive_pairs_train_val_test_1")
train_dataset = dataset["train"]
eval_dataset = dataset["val"]
test_dataset = dataset["test"]

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="/home/ubuntu/Router/models/nomic-embed-text-v1.5-router",
    # Optional training parameters:
    num_train_epochs=30,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    learning_rate=2e-5,
    warmup_ratio=0.01,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    # batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=20,
    logging_steps=100,
    run_name="nomic-embed-text-v1.5-router",  # Will be used in W&B if `wandb` is installed
    report_to="wandb",
    # dataloader_num_workers=0,
    # dataloader_drop_last=True
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = BinaryClassificationEvaluator(
    sentences1=eval_dataset["text1"],
    sentences2=eval_dataset["text2"],
    labels=eval_dataset["label"],
    name="binary-classification-dev",
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 8. Save the trained model
model.save_pretrained("/home/ubuntu/Router/models/nomic-embed-text-v1.5-router/final")

# 9. (Optional) Push it to the Hugging Face Hub
# model.push_to_hub("mpnet-base-all-nli-triplet")