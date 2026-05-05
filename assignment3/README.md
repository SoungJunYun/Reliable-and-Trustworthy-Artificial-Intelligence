# Marabou 신경망 검증 프로젝트 (Assignment #3)

## 1. 설치 방법 (Linux/Colab)
[cite_start]Marabou의 Python API인 `maraboupy`를 사용하기 위해 빌드 과정이 필요합니다. 

# 필수 의존성 설치
sudo apt-get update
sudo apt-get install -y cmake python3-dev python3-pip libboost-all-dev

# Marabou 저장소 클론 및 빌드
git clone [https://github.com/NeuralNetworkVerification/Marabou.git](https://github.com/NeuralNetworkVerification/Marabou.git)

cd Marabou

mkdir build && cd build

cmake .. -DPYTHON_EXECUTABLE=$(which python3)

cmake --build . 

