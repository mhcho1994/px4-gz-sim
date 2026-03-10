import os
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from feature_extractor import extract_from_ulog, extract_from_bin

PX4_BASE_DIR = Path("../../data/px4_logs/")
ARDU_DIR = Path("../../data/ardu_logs/") #need to fix later 

def load_dataset():
    X = []
    y = []
    
    print("Loading PX4 Data...")
    if os.path.exists(PX4_BASE_DIR):
        for file in os.listdir(PX4_BASE_DIR):
            if file.endswith('.ulg'):
                feat = extract_from_ulog(os.path.join(PX4_BASE_DIR, file))
                if feat is not None:
                    X.append(feat)
                    y.append(0)  # PX4 = 0
                    
    print("Loading ArduPilot Data...")
    if os.path.exists(ARDU_DIR):
        for file in os.listdir(ARDU_DIR):
            if file.endswith('.BIN'):
                feat = extract_from_bin(os.path.join(ARDU_DIR, file))
                if feat is not None:
                    X.append(feat)
                    y.append(1)  # ArduPilot = 1
                    
    return np.array(X), np.array(y)

def main():
    X, y = load_dataset()
    
    if len(X) < 5:
        print("\nToo few data points, Check your directory.")
        return
    
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"\nFound Class: {unique_classes}, Each number of Data: {counts}")

    # 학습용 / 테스트용 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ★ 매우 중요: 스케일링 (SVM은 숫자의 크기에 민감하므로 0~1 사이로 정규화 해줌)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SVM 모델 선언 및 학습
    print("\nLearning SVM Model...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train_scaled, y_train)
    
    # 평가
    y_pred = model.predict(X_test_scaled)
    print("\n================ Classified Result ================")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['PX4', 'ArduPilot']))

if __name__ == "__main__":
    main()