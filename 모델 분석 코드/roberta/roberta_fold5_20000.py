import os
import pandas as pd
import torch
import random
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, RobertaTokenizer, RobertaModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix, confusion_matrix
import seaborn as sns
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 사용자 정의 토크나이저 정의
class HexTokenizer:
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = {
                'pad_token': '<pad>',
                'eos_token': '</s>',
                'bos_token': '<s>',
                'unk_token': '<unk>',
                'mask_token': '[MASK]'
            }
        self.special_tokens = special_tokens
        self.token_to_id = {f"{i:02x}": i + len(special_tokens) for i in range(256)}
        self.id_to_token = {i + len(special_tokens): f"{i:02x}" for i in range(256)}

        for idx, token in enumerate(special_tokens.values()):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def tokenize(self, text):
        return [text[i:i+2] for i in range(0, len(text), 2)]

    def encode_plus(self, text_list, max_length, padding='max_length', truncation=True, return_tensors='pt'):
        results = []
        for text in text_list:
            if not text:
                continue

            input_ids = []
            attention_mask = []

            ids = [self.token_to_id[self.special_tokens['bos_token']]]
            ids += [self.token_to_id.get(token, self.token_to_id['<unk>']) for token in self.tokenize(text)]
            ids.append(self.token_to_id[self.special_tokens['eos_token']])

            if len(ids) > max_length and truncation:
                ids = ids[:max_length-1] + [ids[-1]]

            attention = [1] * len(ids)

            if len(ids) < max_length and padding == 'max_length':
                pad_len = max_length - len(ids)
                ids += [self.token_to_id[self.special_tokens['pad_token']]] * pad_len
                attention += [0] * pad_len

            if return_tensors == 'pt':
                result = {
                    'input_ids': torch.tensor([ids], dtype=torch.long),
                    'attention_mask': torch.tensor([attention], dtype=torch.long)
                }
            else:
                result = {
                    'input_ids': ids,
                    'attention_mask': attention
                }
            results.append(result)
        return results

def calculate_app_specific_metrics(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    app_metrics = {}

    for i in range(n_classes):
        app_name = class_names[i]
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / np.sum(cm)

        app_metrics[app_name] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': accuracy
        }

    return app_metrics, cm

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

# 데이터 로드
file_path = '../flow_file/All_random_20000.csv'

if not os.path.exists('./roberta/trained_model_12_20000/matrixs300'):
    os.makedirs('./roberta/trained_model_12_20000/matrixs300')
if not os.path.exists('./roberta/trained_model_12_20000/indexs300'):
    os.makedirs('./roberta/trained_model_12_20000/indexs300')
if not os.path.exists('./roberta/trained_model_12_20000/results300'):
    os.makedirs('./roberta/trained_model_12_20000/results300')
if not os.path.exists('./roberta/trained_model_12_20000/logs300'):
    os.makedirs('./roberta/trained_model_12_20000/logs300')

data = pd.read_csv(file_path)

# 데이터 로드 확인
if data.empty:
    print("Error: No data loaded from the file.")
else:
    print("Data loaded successfully:")
    print(data.head())

# 데이터 필터링
filtered_data = data

# 필터링 결과 확인
if filtered_data.empty:
    print("Error: No data found after filtering by 'Packet Number'.")
else:
    print("Filtered data is available:")
    print(filtered_data.head())

# 공백 데이터 유지 및 페이로드와 어플리케이션 추출
payloads = filtered_data['Payload'].astype(str)
applications = filtered_data['Application'].astype(str)

# 레이블 인코딩
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(applications)
num_classes = len(label_encoder.classes_)

# 클래스 가중치 계산
class_weights = torch.tensor([len(labels) / np.sum(labels == i) for i in range(len(set(labels)))], dtype=torch.float).to(device)

# 토크나이저를 사용하여 토큰화
tokenizer = HexTokenizer()
tokenized_data = tokenizer.encode_plus(payloads.tolist(), max_length=300)

# 사용자 정의 데이터셋 클래스
class PayloadDataset(Dataset):
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input_ids': self.inputs[idx], 'attention_mask': self.masks[idx], 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

# K-Fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

# 교차 검증 수행
for train_index, test_index in kf.split(tokenized_data):
    print(f"\nFold {fold}")

    # 훈련 및 테스트 데이터 분할
    train_inputs = [tokenized_data[i]['input_ids'][0] for i in train_index]
    train_masks = [tokenized_data[i]['attention_mask'][0] for i in train_index]
    train_labels = labels[train_index]

    test_inputs = [tokenized_data[i]['input_ids'][0] for i in test_index]
    test_masks = [tokenized_data[i]['attention_mask'][0] for i in test_index]
    test_labels = labels[test_index]

    # 훈련 데이터를 다시 훈련/검증 데이터로 분할 (75% 훈련, 25% 검증)
    train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = train_test_split(
        train_inputs, train_labels, train_masks, test_size=0.25, random_state=42
    )

    # 데이터셋 생성
    train_dataset = PayloadDataset(train_inputs, train_masks, train_labels)
    val_dataset = PayloadDataset(val_inputs, val_masks, val_labels)
    test_dataset = PayloadDataset(test_inputs, test_masks, test_labels)

    # 모델 로드 및 학습 설정
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base',num_labels=num_classes).to(device)

    # 손실 함수에 클래스 가중치 적용
    from torch.nn import CrossEntropyLoss

    # Trainer 내부에서 손실 함수 설정
    def compute_loss(model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        outputs = model(input_ids = input_ids, attention_mask=attention_mask)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=f'./roberta/trained_model_12_20000/results300/fold_{fold}', num_train_epochs=3, per_device_train_batch_size=4, per_device_eval_batch_size=8,
        gradient_accumulation_steps=8, gradient_checkpointing=False, warmup_steps=600, weight_decay=0.01,
        logging_dir=f'./roberta/trained_model_12_20000/logs300/fold_{fold}', logging_steps=10, evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        fp16=True, # Mixed precision training for speed up
        dataloader_pin_memory=False, # Avoid PyTorch parallel DataLoader warning
        label_smoothing_factor=0.1, # For potential better generalization
        learning_rate=5e-5 # Adjusted learning rate
    )

    # 트레이너 생성
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
        'precision': precision_score(p.label_ids, np.argmax(p.predictions, axis=1), average='macro'),
        'recall': recall_score(p.label_ids, np.argmax(p.predictions, axis=1), average='macro'),
        'f1': f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average='macro')
    },
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

    # 모델 학습
    trainer.train()
    model.save_pretrained(f'./roberta/trained_model_12_20000/model300/fold_{fold}')

    # 테스트 데이터 로더 생성
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    # 모델 테스트 평가
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            true_labels.extend(batch['labels'].cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

    # 전체 성능 메트릭 계산
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')

    print(f"Fold {fold} score:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 어플리케이션별 메트릭 계산 및 혼동 행렬 획득
    class_names = label_encoder.classes_
    app_metrics, cm = calculate_app_specific_metrics(true_labels, pred_labels, class_names)

    # 결과를 DataFrame으로 변환
    app_metrics_df = pd.DataFrame.from_dict(app_metrics, orient='index')
    app_metrics_df.reset_index(inplace=True)
    app_metrics_df.rename(columns={'index': 'Application'}, inplace=True)

    # 결과 저장 및 출력
    app_metrics_df.to_csv(f'./roberta/trained_model_12_20000/results300/fold_{fold}_application_metrics.csv', index=False)
    print("\nApplication-specific metrics:")
    print(app_metrics_df)

    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for Fold {fold}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(f'./roberta/trained_model_12_20000/matrixs300/fold_{fold}_confusion_matrix.png')
    plt.close()

    # 데이터 저장 함수
    def save_data(file_path, inputs, masks, labels):
        data = {
            'input_ids': [ids.tolist() for ids in inputs],
            'attention_mask': [mask.tolist() for mask in masks],
            'labels': labels.tolist()
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    # 훈련, 검증, 테스트 데이터 저장
    save_data(f'./roberta/trained_model_12_20000/indexs300/fold_{fold}_train_data.csv', train_inputs, train_masks, train_labels)
    save_data(f'./roberta/trained_model_12_20000/indexs300/fold_{fold}_val_data.csv', val_inputs, val_masks, val_labels)
    save_data(f'./roberta/trained_model_12_20000/indexs300/fold_{fold}_test_data.csv', test_inputs, test_masks, test_labels)

    fold += 1
