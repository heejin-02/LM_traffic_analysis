import pandas as pd

# fold 5 예측 결과 로드
fold = 5
predictions_path = f'./roberta/trained_model_12_1000/results/fold_{fold}_predictions.csv'
predictions_df = pd.read_csv(predictions_path)

# 실제 레이블과 예측된 레이블 가져오기
true_labels = predictions_df['true_labels'].tolist()
pred_labels = predictions_df['pred_labels'].tolist()

# 잘못된 분류된 인덱스 찾기
incorrect_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) if true != pred]

# 잘못 분류된 페이로드를 포함한 테스트 데이터 로드
test_data_path = f'./roberta/trained_model_12_1000/indexs/fold_{fold}_test_data.csv'
test_data = pd.read_csv(test_data_path)

# 헥사코드를 아스키 문자열로 변환하는 함수
def hex_to_ascii(hex_list):
    ascii_str = ""
    for hex_code in hex_list:
        if hex_code in ['<s>', '</s>', '<pad>']:  # 특수 토큰은 그대로 유지
            ascii_str += f" {hex_code} "
        else:
            try:
                ascii_str += bytes.fromhex(hex_code).decode('ascii')
            except ValueError:
                ascii_str += '.'  # 아스키 변환이 불가능한 경우 대체 문자 사용
    return ascii_str

# 잘못 분류된 페이로드 정보 저장 및 출력
incorrect_data = []

for idx in incorrect_indices:
    payload_hex = eval(test_data['input_ids'][idx])  # 문자열로 저장된 리스트를 평가하여 리스트로 변환
    true_label = true_labels[idx]
    predicted_label = pred_labels[idx]

    # 헥사 페이로드를 아스키 문자열로 변환
    payload_ascii = hex_to_ascii(payload_hex)

    # 잘못 분류된 데이터를 리스트에 추가
    incorrect_data.append({
        'Index': idx,
        'True Label': true_label,
        'Predicted Label': predicted_label,
        'Payload (Hex)': payload_hex,
        'Payload (ASCII)': payload_ascii
    })

    # 잘못 분류된 페이로드 출력
    print(f"Index: {idx}, True Label: {true_label}, Predicted: {predicted_label}")
    print(f"Payload (Hex): {payload_hex}")
    print(f"Payload (ASCII): {payload_ascii}\n")

# 잘못 분류된 데이터를 데이터프레임으로 변환 후 CSV 파일로 저장
incorrect_data_df = pd.DataFrame(incorrect_data)
output_path = f'./roberta/trained_model_12_20000/results/fold_{fold}_misclassified_payloads.csv'
incorrect_data_df.to_csv(output_path, index=False)

print(f"Misclassified payloads saved to {output_path}")
