import sys
import os
import numpy as np

# Marabou 빌드 경로 설정 (Colab 환경 기준)
MARABOU_DIR = "/content/Marabou"
sys.path.append(MARABOU_DIR)

from maraboupy import Marabou

def run_verification():
    """
    Marabou를 사용하여 외부 ONNX 모델의 강건성을 검증하는 함수입니다.
    """
    # 1. 모델 로드
    model_path = "tiny_mnist.onnx"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} 파일을 찾을 수 없습니다. 모델 생성 코드를 먼저 실행하세요.")
        return

    # ONNX 모델 읽기
    network = Marabou.read_onnx(model_path)

    # 2. 입력 제약 조건 설정 (Robustness 쿼리)
    # network.inputVars[0]는 보통 (1, 784) 형태의 리스트입니다.
    input_vars = network.inputVars[0].flatten() 
    epsilon = 0.02 

    for var_index in input_vars:
        # 안전한 인덱스 참조를 위해 int로 변환
        v = int(var_index)
        network.setLowerBound(v, -epsilon)
        network.setUpperBound(v, epsilon)

    # 3. 출력 제약 조건 설정
    # 클래스 0의 출력값이 0.5보다 큰 경우가 존재하는지 확인 (Property)
    output_vars = network.outputVars[0].flatten()
    target_output_var = int(output_vars[0])
    
    # Inequality: 1 * output_0 >= 0.5
    network.addInequality([target_output_var], [1], 0.5) 

    # 4. 검증 실행 및 결과 해석
    print(f"--- Marabou Verification Start (Model: {model_path}) ---")
    
    # solve() 호출
    exit_code, vals, stats = network.solve()
    
    print("\n" + "="*30)
    if exit_code == "unsat":
        print("결과: UNSAT")
        print("의미: 설정한 범위 내에서 클래스 0이 0.5를 넘는 경우는 절대 없습니다. (안전 증명)")
    elif exit_code == "sat":
        print("결과: SAT")
        print("의미: 조건에 맞는 반례(Counter-example)를 찾았습니다. 모델이 취약할 수 있습니다.")
    else:
        print(f"결과: {exit_code} (검증 중단 또는 알 수 없는 상태)")

    # stats 메서드 명칭 수정 반영: getTotalTimeInMicroseconds -> getTotalTimeInMicro
    try:
        total_time_ms = stats.getTotalTimeInMicro() / 1000
    except AttributeError:
        # 혹시 구버전 라이브러리일 경우를 대비한 예외 처리
        total_time_ms = stats.getTotalTimeInMicroseconds() / 1000
        
    print(f"검증 소요 시간: {total_time_ms:.2f} ms")
    print("="*30)

if __name__ == "__main__":
    run_verification()