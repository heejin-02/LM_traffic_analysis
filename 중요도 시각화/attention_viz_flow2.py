import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_model(model_path, device='cuda'):
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def hex_to_readable(token):
    """
    헥사코드를 사람이 읽을 수 있는 텍스트로 변환.
    ASCII 문자로 변환하고, 변환 불가능한 경우 '.'으로 표시합니다.
    """
    try:
        char = bytes.fromhex(token).decode('ascii')
    except (ValueError, UnicodeDecodeError):
        char = '.'  # 변환 불가능한 경우 '.'으로 대체
    return char

def visualize_token_importance(model, tokenizer, token_ids, label, flow_index, output_dir, device='cuda', importance_threshold=0.001, num_tokens_to_display=50):
    """
    모델의 중요도를 계산하고 시각화합니다.
    """
    model.eval()

    encoded = tokenizer.encode_plus(token_ids, max_length=200, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        original_prob = torch.softmax(output.logits, dim=1)[0, label].item()

    importance = []
    for i in range(input_ids.shape[1]):
        if attention_mask[0, i] == 0:
            importance.append(0)
            continue

        masked_input_ids = input_ids.clone()
        masked_input_ids[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            masked_output = model(masked_input_ids, attention_mask=attention_mask)
            masked_prob = torch.softmax(masked_output.logits, dim=1)[0, label].item()

        importance.append(original_prob - masked_prob)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

    # 헥사코드를 사람이 읽을 수 있는 텍스트로 변환
    readable_tokens = [hex_to_readable(token) for token in tokens]

    # 중요도가 임계값 이상인 토큰만 표시
    displayed_importance = [imp if imp >= importance_threshold else 0 for imp in importance]

    # 앞부분의 토큰만 선택
    displayed_tokens = tokens[:num_tokens_to_display]
    displayed_readable_tokens = readable_tokens[:num_tokens_to_display]
    displayed_importance = displayed_importance[:num_tokens_to_display]

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 시각화
    plt.figure(figsize=(20, 5))
    sns.heatmap([displayed_importance], cmap='YlOrRd', annot=[displayed_readable_tokens], fmt='', cbar=True,
                annot_kws={'rotation': 90, 'fontsize': 8},
                yticklabels=False, xticklabels=False)
    plt.title(f'Token Importance for label {label}')
    plt.xlabel('Token Position')
    plt.tight_layout()

    # 이미지 파일명을 flow_index를 사용하여 번호를 매김
    plt.savefig(os.path.join(output_dir, f'token_importance_visualization_{flow_index}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    """
    메인 실행 함수. 모델과 토크나이저를 로드하고, CSV 데이터를 처리합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 및 토크나이저 로드
    tokenizer_path = './trained_model_12_1000/tokenizer'
    model_path = './trained_model_12_1000/models1/fold_1'
    model = load_model(model_path, device)
    tokenizer = load_tokenizer(tokenizer_path)

    # CSV 데이터 로드
    csv_path = './trained_model_12_1000/indexs200/fold_1_test_data.csv'
    df = load_csv_data(csv_path)

    # 출력 폴더 설정
    output_dir = './token_importance_visualizations'

    # 처음 500개 행만 처리
    for index, row in df.head(500).iterrows():
        input_ids = row['input_ids']
        label = int(row['labels'])  # 필요한 경우 정수형으로 변환

        # input_ids를 리스트로 변환
        token_ids = eval(input_ids)
        
        # 200개의 토큰 단위로 플로우를 분할
        flow_count = 0
        for i in range(0, len(token_ids), 200):
            flow_token_ids = token_ids[i:i+200]
            
            # 토큰의 길이가 200이 되지 않으면 패딩 처리
            if len(flow_token_ids) < 200:
                flow_token_ids += [tokenizer.pad_token_id] * (200 - len(flow_token_ids))
            
            # 시각화
            flow_index = index * (len(token_ids) // 200) + flow_count + 1
            print(f"Visualizing token importance for row {index}, flow {flow_index}")
            visualize_token_importance(model, tokenizer, flow_token_ids, label, flow_index, output_dir, device, num_tokens_to_display=50)
            flow_count += 1

if __name__ == "__main__":
    main()
