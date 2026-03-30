import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import DroneTrajectoryCNN
from dataset import build_training_pipeline

def analyze_feature_importance():
    """각 특성이 분류에 미치는 영향 분석"""
    
    # 데이터 로드
    train_loader, test_loader = build_training_pipeline(
        px4_dir="../../data/px4_logs", 
        ardu_dir="../../data/ardu_logs", 
        batch_size=32, 
        test_ratio=0.2,
        window_size=100,
        step_size=50
    )

    # 모델 로드
    model = DroneTrajectoryCNN(num_features=7)
    model.load_state_dict(torch.load("drone_real_model.pth"))
    model.eval()

    feature_names = ['Speed', 'Accel Mag', 'Jerk Mag', 'Curvature', 'Yaw Rate (Traj)', 'Yaw Rate (Att)', 'Slip Rate']

    print(f"\n{'='*70}")
    print("🔍 FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*70}\n")

    # 1. 각 특성을 0으로 masking했을 때 정확도 변화 측정
    print("📊 Permutation Importance (각 특성을 제거했을 때 정확도 변화)\n")
    
    # 기본 정확도 계산
    baseline_correct = 0
    baseline_total = 0
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            all_outputs.append(outputs)
            all_targets.append(batch_y)
            _, predicted = torch.max(outputs.data, 1)
            baseline_total += batch_y.size(0)
            baseline_correct += (predicted == batch_y).sum().item()
    
    baseline_acc = baseline_correct / baseline_total
    print(f"Baseline Accuracy: {baseline_acc*100:.2f}%\n")
    
    # 각 특성별 중요도 계산
    importance_scores = []
    
    for feature_idx in range(7):
        feature_drop_correct = 0
        feature_drop_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # 해당 특성을 0으로 masking
                masked_batch = batch_x.clone()
                masked_batch[:, feature_idx, :] = 0
                
                outputs = model(masked_batch)
                _, predicted = torch.max(outputs.data, 1)
                feature_drop_total += batch_y.size(0)
                feature_drop_correct += (predicted == batch_y).sum().item()
        
        feature_drop_acc = feature_drop_correct / feature_drop_total
        importance = baseline_acc - feature_drop_acc  # 양수면 중요함
        importance_scores.append(importance * 100)  # 퍼센트로 변환
        
        direction = "↓" if importance > 0 else "↑"
        print(f"{feature_idx+1}. {feature_names[feature_idx]:20s}: {importance*100:+.2f}% {direction}")
    
    # 정렬
    sorted_indices = np.argsort(importance_scores)[::-1]
    
    print(f"\n{'─'*70}")
    print("중요도 순서 (높음 → 낮음):\n")
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"  {rank}. {feature_names[idx]:20s}: {importance_scores[idx]:+.2f}%")
    
    # 2. Conv1d 가중치 분석
    print(f"\n{'─'*70}")
    print("📈 First Conv1d Layer Weight Magnitude Analysis\n")
    
    conv1_weights = model.conv1.weight.data  # (out_channels, in_channels, kernel_size)
    # 각 입력 특성별로 가중치의 절댓값 평균
    feature_magnitude = np.mean(np.abs(conv1_weights.cpu().numpy()), axis=(0, 2))
    
    print("Average weight magnitude per feature in Conv1d:")
    for i, name in enumerate(feature_names):
        print(f"  {name:20s}: {feature_magnitude[i]:.4f}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Permutation Importance
    colors = ['green' if x > 0 else 'red' for x in importance_scores]
    axes[0].barh(feature_names, importance_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Importance (Accuracy Drop %)')
    axes[0].set_title('Permutation Importance of Features')
    axes[0].axvline(0, color='black', linestyle='-', linewidth=0.8)
    axes[0].grid(alpha=0.3)
    
    # Weight Magnitude
    axes[1].bar(feature_names, feature_magnitude, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Average Weight Magnitude')
    axes[1].set_title('Conv1d Weight Magnitude Analysis')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("\n✅ Feature importance visualization saved to 'feature_importance.png'")
    
    # 해석
    print(f"\n{'='*70}")
    print("💡 INTERPRETATION")
    print(f"{'='*70}")
    
    top_feature = feature_names[sorted_indices[0]]
    top_importance = importance_scores[sorted_indices[0]]
    
    print(f"\n✨ Most important feature: {top_feature}")
    print(f"   - Removing this feature drops accuracy by {abs(top_importance):.2f}%")
    
    if top_importance > 0:
        print(f"   - This suggests that PX4 and ArduPilot have different {top_feature.lower()}")
    else:
        print(f"   - Removing this feature IMPROVED accuracy - it might be noise!")
    
if __name__ == "__main__":
    analyze_feature_importance()
