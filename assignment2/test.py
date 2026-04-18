import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# 1. 환경 설정 및 데이터 로드
def setup_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    return x_test[:100], y_test[:100]  # 테스트를 위해 100개 샘플 사용 

# 2. ResNet50 모델 구성 (Problem 1) [cite: 34]
def build_model(name, weights='imagenet'):
    base = ResNet50(weights=weights, include_top=False, input_shape=(32, 32, 3))
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(10, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output, name=name)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. 뉴런 커버리지 계산 함수 
def get_neuron_coverage(model, input_data, threshold=0.5):
    # 마지막 컨볼루션 레이어의 활성화를 관찰 (DeepXplore 핵심 지표)
    layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    activations = intermediate_model.predict(input_data)
    # 활성화 값이 문턱값(threshold)을 넘은 뉴런 비율 계산
    activated = np.sum(np.mean(activations, axis=0) > threshold)
    total = activations.shape[-1]
    return activated / total

# 4. 차분 테스팅 실행 (Problem 1) 
def run_differential_testing(models, data, labels):
    if not os.path.exists('results'):
        os.makedirs('results')

    preds1 = np.argmax(models[0].predict(data), axis=1)
    preds2 = np.argmax(models[1].predict(data), axis=1)
    
    # 두 모델의 예측이 다른 인덱스 추출 
    disagreement_idx = np.where(preds1 != preds2)[0]
    
    print(f"Total Disagreements Found: {len(disagreement_idx)}") 
    
    # 상위 5개 불일치 사례 시각화 및 저장 
    for i in disagreement_idx[:5]:
        plt.figure(figsize=(4,4))
        plt.imshow(data[i])
        plt.title(f"M1: {preds1[i]} vs M2: {preds2[i]} (True: {labels[i][0]})")
        plt.axis('off')
        plt.savefig(f'results/disagreement_{i}.png')
        plt.close()
    
    return len(disagreement_idx)

if __name__ == "__main__":
    test_images, test_labels = setup_data()
    
    # 서로 다른 가중치로 설정된 두 모델 로드 
    model1 = build_model("ResNet50_V1", weights='imagenet')
    model2 = build_model("ResNet50_V2", weights=None) # 무작위 초기화 모델
    
    # 테스팅 수행
    num_diff = run_differential_testing([model1, model2], test_images, test_labels)
    cov = get_neuron_coverage(model1, test_images)
    
    print("-" * 30)
    print(f"Final Neuron Coverage: {cov:.2%}") 
    print(f"Images saved in results/ directory.") 