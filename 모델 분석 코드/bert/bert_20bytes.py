import pandas as pd
import torch
import random
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import numpy as np

# 사용자 정의 토크나이저 정의
class HexTokenizer:
    def __init__(self, delimiter=':', special_tokens=None):
        self.delimiter = delimiter
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
        return text.split(self.delimiter)

    def mask_tokens(self, ids):
        output_ids = []
        for id_list in ids:
            new_ids = []
            for id in id_list:
                if random.random() < 0.15:
                    rand = random.random()
                    if rand < 0.8:
                        new_ids.append(self.token_to_id['[MASK]'])
                    elif rand < 0.9:
                        new_ids.append(random.choice(list(self.token_to_id.values())))
                    else:
                        new_ids.append(id)
                else:
                    new_ids.append(id)
            output_ids.append(new_ids)
        return output_ids

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

            ids = self.mask_tokens([ids])[0]

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

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

# 데이터 로드
file_path = './TCP_combined_.csv'
data = pd.read_csv(file_path)

# 데이터 로드 확인
if data.empty:
    print("Error: No data loaded from the file.")
else:
    print("Data loaded successfully:")
    print(data.head())

# 각 플로우의 패킷(패킷 번호가 0, 1, 2, 3) 데이터 필터링
filtered_data = data[data['Packet Number'].isin([0, 1, 2, 3])]

# 필터링 결과 확인
if filtered_data.empty:
    print("Error: No data found after filtering by 'Packet Number'.")
else:
    print("Filtered data is available:")
    print(filtered_data.head())

# 앞부분 20바이트만 사용하여 페이로드와 어플리케이션 추출
payloads = filtered_data['Payload'].astype(str).apply(lambda x: x[:20*2])  # 20 bytes -> 40 hex characters
applications = filtered_data['Application'].astype(str)

# 레이블 인코딩
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(applications)

# 클래스 가중치 계산
class_weights = torch.tensor([len(labels) / np.sum(labels == i) for i in range(len(set(labels)))], dtype=torch.float).to(device)

# 토크나이저를 사용하여 토큰화
tokenizer = HexTokenizer()
tokenized_data = tokenizer.encode_plus(payloads.tolist(), max_length=512)

# 데이터셋 분할
train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = train_test_split(
    [item['input_ids'].squeeze() for item in tokenized_data], labels,
    [item['attention_mask'].squeeze() for item in tokenized_data], test_size=0.1, random_state=42
)

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

# 데이터셋 생성
train_dataset = PayloadDataset(train_inputs, train_masks, train_labels)
val_dataset = PayloadDataset(val_inputs, val_masks, val_labels)

# 모델 로드 및 학습 설정
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set(labels)), force_download=True).to(device)

# 손실 함수에 클래스 가중치 적용
from torch.nn import CrossEntropyLoss

# Trainer 내부에서 손실 함수 설정
def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")
    loss_fct = CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='./results_test9', num_train_epochs=5, per_device_train_batch_size=4, per_device_eval_batch_size=8,
    gradient_accumulation_steps=8, gradient_checkpointing=False, warmup_steps=600, weight_decay=0.01,
    logging_dir='./logs_test9', logging_steps=10, evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    fp16=True, # Mixed precision training for speed up
    dataloader_pin_memory=False, # Avoid PyTorch parallel DataLoader warning
    label_smoothing_factor=0.1, # For potential better generalization
    learning_rate=5e-5 # Adjusted learning rate
)

# 트레이너 생성
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
        'precision': precision_score(p.label_ids, np.argmax(p.predictions, axis=1), average='macro'),
        'recall': recall_score(p.label_ids, np.argmax(p.predictions, axis=1), average='macro'),
        'f1': f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average='macro')
    },
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # Early stopping
)

# 모델 학습
trainer.train()
model.save_pretrained('./trained_model_test14')

# 검증 데이터 로더 생성
val_dataloader = DataLoader(val_dataset, batch_size=16)

# 모델 평가
model.eval()
true_labels, pred_labels = [], []
with torch.no_grad():
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        true_labels.extend(batch['labels'].cpu().numpy())
        pred_labels.extend(predictions.cpu().numpy())

# 평가 지표 계산
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='macro')
recall = recall_score(true_labels, pred_labels, average='macro')
f1 = f1_score(true_labels, pred_labels, average='macro')
print(f"Accuracy: {accuracy:.4f} \nPrecision: {precision:.4f} \nRecall: {recall:.4f} \nF1 Score: {f1:.4f}")
