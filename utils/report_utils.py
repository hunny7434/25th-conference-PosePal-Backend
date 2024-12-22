from utils.test import *
from utils.model.model import *
import pickle
import os


model_path = "utils/model/lateralraise_fin.pkl"  # 모델 경로 설정
        
def run_posture_model(video_path):
    # Placeholder: 실제 모델을 로드하고 비디오를 분석하는 로직을 구현하세요.
    # 예시로 간단한 문자열을 반환합니다.
    smoothed_data = process_video_and_smooth(video_path)
    segments = segment_reps(smoothed_data)
    input_data = combine_segments(segments)
    
    predicted_label = inference(model_path, input_data)
    
    feedback_report = "video_path: " + video_path + "\n운동 자세가 전반적으로 양호합니다. 약간의 개선이 필요한 부분은 다음과 같습니다: 팔과 다리의 정렬을 더 신경 써주세요."
    return feedback_report, predicted_label