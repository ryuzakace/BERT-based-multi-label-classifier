import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")

    label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    dfx['labels'] = dfx[label_columns].apply(lambda x: list(x), axis=1)
    dfx.drop(['id'], inplace=True, axis=1)
    dfx.drop(label_columns, inplace=True, axis=1)

    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.labels.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        comment_text=df_train.comment_text.values, target=df_train.labels.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        comment_text=df_valid.comment_text.values, target=df_valid.labels.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device(config.DEVICE)
    model = BERTBaseUncased()
    model.to(device)

    # param_optimizer = list(model.named_parameters())
    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # optimizer_parameters = [
    #     {
    #         "params": [
    #             p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay": 0.001,
    #     },
    #     {
    #         "params": [
    #             p for n, p in param_optimizer if any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay": 0.0,
    #     },
    # ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)

        #TODO define accuracy calculation for multi label
        outputs = np.array(outputs) >= 0.5 #here maybe scope for hyperparameters?
        accuracy = metrics.accuracy_score([i for j in targets for i in j], [i for j in outputs for i in j])
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()
