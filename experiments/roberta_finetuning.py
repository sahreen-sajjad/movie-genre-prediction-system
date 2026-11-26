"""Historical notebook experiment; not independently validated."""
from src.data import load_data, split_data

def main():
    import pandas as pd
    final_df = load_data()
    train, test, val = split_data(final_df)
    #to save the original unchanged
    BERT_train = train.copy()
    BERT_val = val.copy()
    BERT_test = test.copy()

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    #evaluation function to print metrics

    def evaluate(val, val_predictions):
        val_precision, val_recall, val_f1, support = precision_recall_fscore_support(val['meta.genre'], val_predictions, average='macro', zero_division=True)
        val_accuracy = (val['meta.genre'] == val_predictions).mean()
        print(f'Validation accuracy: {val_accuracy:.3f}')
        print(f'Validation precision (macro-averaged): {val_precision:.3f}')
        print(f'Validation recall (macro-averaged): {val_recall:.3f}')
        print(f'Validation F1-score (macro-averaged): {val_f1:.3f}')

    from transformers import RobertaModel, RobertaTokenizer

    from datasets import Dataset
    import numpy as np
    from transformers import RobertaForSequenceClassification
    from transformers import RobertaTokenizer
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, Trainer, TrainingArguments
    from transformers import DataCollatorForTokenClassification
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    np.random.seed(42)

    import transformers

    #roberta model and tokeniser
    from transformers import RobertaForSequenceClassification
    num_labels = 6
    model = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=num_labels)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


    #return values for labels
    def get_tag_id(tag):
        tag_id_map = {'drama': 0, 'horror': 1, 'thriller': 2, 'comedy': 3, 'documentary': 4}
        return tag_id_map.get(tag, 5)

    #encode text
    train_encodings = tokenizer(list(BERT_train['text']), truncation=True, padding=True)
    val_encodings = tokenizer(list(BERT_val['text']), truncation=True, padding=True)

    y_Btrain = [get_tag_id(tag) for tag in BERT_train['meta.genre']]
    y_Btrain = np.array(y_Btrain)

    y_Bval = [get_tag_id(tag) for tag in BERT_val['meta.genre']]
    y_Bval = np.array(y_Bval)

    train_dataset = [{'input_ids': train_encodings['input_ids'][i], 
                      'attention_mask': train_encodings['attention_mask'][i], 
                      'labels': y_Btrain[i]} for i in range(len(BERT_train))]
    val_dataset = [{'input_ids': val_encodings['input_ids'][i], 
                    'attention_mask': val_encodings['attention_mask'][i], 
                    'labels': y_Bval[i]} for i in range(len(BERT_val))]

    train_features = {
        'input_ids': np.array([f['input_ids'] for f in train_dataset]), 
        'attention_mask': np.array([f['attention_mask'] for f in train_dataset]),
        'labels': np.array([f['labels'] for f in train_dataset])
    }


    val_features = {'input_ids': np.array([f['input_ids'] for f in val_dataset]), 
                    'attention_mask': np.array([f['attention_mask'] for f in val_dataset]),
                    'labels': np.array([f['labels'] for f in val_dataset])}


    train_dataset = Dataset.from_dict(train_features)
    val_dataset = Dataset.from_dict(val_features)

    from transformers import Trainer

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0,
        max_steps=5,
        logging_steps = 700
    )

    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    import numpy as np
    def compute_metrics(pred):
        predictions, genre_labels = pred
        preds_flat = np.argmax(predictions, axis=1).flatten()
        labels_flat = genre_labels.flatten()
        indices = np.where(labels_flat != -100)
        preds = preds_flat[indices]
        true_labels = labels_flat[indices]
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')
        accuracy = accuracy_score(true_labels, preds)
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


    trainer.train()

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0,
        max_steps=5,
        logging_steps = 700
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


    trainer.train()

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=5e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=0,
        weight_decay=0,
        max_steps=5,
        logging_steps = 700
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0,
        max_steps=5,
        logging_steps = 700
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    main()
