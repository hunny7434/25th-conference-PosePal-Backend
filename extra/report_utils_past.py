import cv2
import os
import math
import csv
import pandas as pd
import numpy as np
import mediapipe as mp
import streamlit as st
import tempfile
import re
from scipy.signal import find_peaks, savgol_filter

############################################### 1. Process Frames ###############################################
def get_frames(video_path, desired_fps=30):
    """
    동영상에 프레임 추출 및 저장.

    매개변수:
        video_path (str): 입력 동영상 파일의 경로.
        output_dir (str): 추출된 프레임이 저장될 디렉토리, current: {video_path}/test/frames
        desired_fps (int): 초당 추출할 프레임 수.
    반환값:
    """
    # 출력 디렉토리를 비디오 경로 하위에 생성
    output_dir = f"{video_path}/test/frames"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = math.floor(original_fps / desired_fps) if original_fps > 0 else 1

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 'frame_interval' 간격으로 프레임 저장
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.png")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1
        
    cap.release()


############################################### 2. Pose Estimation ###############################################
def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)', filename)
    return int(match.group(1)) if match else None

def save_pose_landmarks(frame_no, results, output_csv, selected_landmarks):
    try:
        keypoints = [frame_no]
        for landmark_id in selected_landmarks:
            landmark = results.pose_landmarks.landmark[landmark_id]
            keypoints.extend([landmark.x, landmark.y, landmark.z])

        # 데이터 CSV에 추가
        with open(output_csv, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(keypoints)
    except Exception as e:
        print(f"Error saving landmarks for frame {frame_no}: {e}")

def pose_estimate(input_dir, video_path, exercise):
    """
    Mediapipe를 사용하여 포즈 추정 및 데이터를 처리합니다.

    output_pose_images_dir = f"{video_path}/test/frames_estimated"
    output_csv = f"{video_path}/test/stats/pose_estimation.csv"
    매개변수:
        input_dir (str): 원본 이미지가 포함된 디렉토리.
        video_path (str): 비디오 경로를 기준으로 결과 이미지를 저장할 디렉토리 생성.
        exercise (str): 처리할 운동 유형.
    """
    # Mediapipe Pose 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    output_pose_images_dir = f"{video_path}/test/frames_estimated"
    output_csv = f"{video_path}/test/stats/pose_estimation.csv"
    os.makedirs(output_pose_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    selected_landmarks_mapping = {
        "sideLateralRaise": [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_PINKY,
            mp_pose.PoseLandmark.RIGHT_PINKY,
            mp_pose.PoseLandmark.LEFT_INDEX,
            mp_pose.PoseLandmark.RIGHT_INDEX,
            mp_pose.PoseLandmark.LEFT_THUMB,
            mp_pose.PoseLandmark.RIGHT_THUMB,
        ]
    }

    if exercise not in selected_landmarks_mapping:
        raise ValueError(f"Exercise '{exercise}' not supported.")

    selected_landmarks = selected_landmarks_mapping[exercise]

    # 선택한 포즈에 대한 랜드마크
    landmarks = ['frame_no']
    for landmark in selected_landmarks:
        landmarks += [f'{landmark.name}_x', f'{landmark.name}_y', f'{landmark.name}_z']

    # CSV 파일 생성 및 헤더 작성
    with open(output_csv, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(landmarks)

    # 이미지 처리 및 랜드마크 저장
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_files = [file for file in os.listdir(input_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
        frame_files = sorted(frame_files, key=lambda x: extract_frame_number(x))

        for file in frame_files:
            frame_no = extract_frame_number(file)
            if frame_no is None:
                print(f"Skipping file with invalid frame format: {file}")
                continue

            file_path = os.path.join(input_dir, file)

            # 이미지 읽기 및 처리
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 포즈 추정 수행
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                save_pose_landmarks(frame_no, results, output_csv, selected_landmarks)

                # 이미지에 포즈 주석 추가
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

                # 포즈 주석 이미지 저장
                output_image_path = os.path.join(output_pose_images_dir, file)
                cv2.imwrite(output_image_path, annotated_image)

            print(f"Processed: Frame {frame_no}")

    # CSV 로드 및 시간 시리즈 정렬
    data = pd.read_csv(output_csv)
    data = data.sort_values(by=['frame_no']).reset_index(drop=True)
    data.to_csv(output_csv, index=False)

    print(f"Pose estimation and time-series data saved successfully to {output_csv}.")
    
############################################### 3. Data Smoothing ###############################################
def smooth_timeseries(file_path, video_path, window_length=31, polyorder=3):
    """
    시간 시리즈 데이터를 매끄럽게 처리하고 새 CSV 파일에 저장.

    매개변수:
        file_path (str): 원본 CSV 파일 경로.
        video_path (str): 비디오 경로를 기준으로 매끄럽게 처리된 데이터를 저장할 디렉토리 생성.
        window_length (int): Savitzky-Golay 필터의 창 길이.
        polyorder (int): Savitzky-Golay 필터의 다항식 차수.
    """
    # 매끄럽게 처리된 CSV 저장 경로
    smoothed_csv = f"{video_path}/test/stats/pose_estimation_smoothed.csv"
    os.makedirs(os.path.dirname(smoothed_csv), exist_ok=True)

    # CSV 파일 로드
    data = pd.read_csv(file_path)

    # 키포인트 및 좌표 열 필터링
    coordinate_columns = [col for col in data.columns if "_" in col and col != 'frame_no']

    # 매끄럽게 처리된 데이터 저장
    smoothed_data = data.copy()
    for col in coordinate_columns:
        time_series = data[col].dropna().values
        if len(time_series) > window_length:  # 데이터가 충분히 있는 경우에만 처리
            smoothed_values = savgol_filter(time_series, window_length=window_length, polyorder=polyorder)
            smoothed_data[col] = smoothed_values

    # 새 CSV에 저장
    smoothed_data.to_csv(smoothed_csv, index=False)
    print(f"Smoothed data saved to: {smoothed_csv}")
    
############################################### 4. Peak Detection and Segmentation ###############################################
def detect_peaks_and_troughs(smoothed_series):
    """
    피크 및 트로프를 감지합니다.

    매개변수:
        smoothed_series (array-like): 매끄럽게 처리된 시간 시리즈 데이터.

    반환값:
        array: 트로프 인덱스.v
    """
    # Detect raw peaks and troughs
    raw_peaks, _ = find_peaks(smoothed_series, height=0, distance=5)
    raw_troughs, _ = find_peaks(-smoothed_series, distance=5)

    # Custom detection for near-zero troughs
    zero_threshold = 0.01  # Adjust based on the data range
    zero_troughs = np.where(np.abs(smoothed_series) < zero_threshold)[0]

    # Include the first trough if it exists
    if smoothed_series[0] < smoothed_series[1]:  # Check if the first point is a trough
        zero_troughs = np.append(zero_troughs, 0)

    # Combine and deduplicate trough indices
    all_troughs = sorted(set(raw_troughs).union(zero_troughs))

    # Refine extrema detection
    refined_troughs = []
    for trough in all_troughs:
        if 0 <= trough < len(smoothed_series) - 1:  # Include the first trough explicitly
            if (trough == 0 or smoothed_series[trough] < smoothed_series[trough - 1]) and \
               (trough == len(smoothed_series) - 1 or smoothed_series[trough] < smoothed_series[trough + 1]):
                refined_troughs.append(trough)

    return refined_troughs

def segment_reps(input_file, video_path):
    """
    피크 감지 및 세그먼트 추출을 수행하고 결과를 CSV 파일로 저장합니다.

    매개변수:
        input_file (str): 매끄럽게 처리된 시간 시리즈 CSV 파일 경로.
        video_path (str): 비디오 경로를 기준으로 결과 세그먼트를 저장할 디렉토리 생성.
    """
    output_segments_dir = f"{video_path}/test/stats/segments"
    os.makedirs(output_segments_dir, exist_ok=True)

    # CSV 파일 로드
    data = pd.read_csv(input_file)

    # LEFT_ELBOW 열을 참조 열로 사용
    reference_column = "LEFT_ELBOW_y"
    if reference_column not in data.columns:
        raise ValueError(f"Reference column '{reference_column}' not found in the dataset.")

    reference_series = data[reference_column].dropna().values
    reference_troughs = detect_peaks_and_troughs(reference_series)

    # 세그먼트 생성
    if len(reference_troughs) > 1:
        for i in range(len(reference_troughs) - 1):
            start = reference_troughs[i]
            end = reference_troughs[i + 1]

            # 세그먼트 데이터 추출
            segment = data.iloc[start:end + 1]

            # 세그먼트를 하나의 CSV 파일로 저장
            segment_file_name = f"rep_segment_{i + 1}.csv"
            segment.to_csv(os.path.join(output_segments_dir, segment_file_name), index=False)

            print(f"Segment {i + 1} saved: {segment_file_name}")
    else:
        print("Not enough troughs detected to extract segments.")
        

############################################### 5. Extract Frames ###############################################
def extract_frames(segments_dir, output_dir):
    """
    각 세그먼트에서 대표 프레임 5개를 선택하여 저장합니다.

    매개변수:
        segments_dir (str): 세그먼트 파일이 저장된 디렉토리 경로.
        output_dir (str): 선택된 프레임을 저장할 디렉토리 경로.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get list of segment files
    segment_files = [f for f in os.listdir(segments_dir) if f.startswith("rep_segment_") and f.endswith(".csv")]

    for segment_file in segment_files:
        segment_path = os.path.join(segments_dir, segment_file)
        segment_data = pd.read_csv(segment_path)

        # Extract keypoint-coordinate columns
        keypoint_columns = [col for col in segment_data.columns if "_" in col and col != "frame_no"]

        # Select meaningful frames for the entire segment: start, intermediate points, peaks, and end
        start_idx = 0
        end_idx = len(segment_data) - 1

        # Detect peaks within the segment based on the average of all keypoints
        avg_series = segment_data[keypoint_columns].mean(axis=1).values
        peaks, _ = find_peaks(avg_series)

        # Ensure exactly 5 selected rows: start, midpoint between start and peak, peak, midpoint between peak and end, and end
        selected_indices = [start_idx]

        if len(peaks) > 0:
            # Use the first peak as reference
            peak_idx = peaks[0]
            selected_indices.append(peak_idx)

            # Add intermediate points
            mid_start_peak_idx = (start_idx + peak_idx) // 2
            mid_peak_end_idx = (peak_idx + end_idx) // 2

            selected_indices.extend([mid_start_peak_idx, mid_peak_end_idx])
        else:
            # If no peak is found, add midpoint
            midpoint_idx = (start_idx + end_idx) // 2
            selected_indices.append(midpoint_idx)

        selected_indices.append(end_idx)

        # Ensure unique indices and sort them
        selected_indices = sorted(set(selected_indices))

        # Collect the selected rows
        selected_data = segment_data.iloc[selected_indices]

        # Save the selected rows to a new CSV file
        selected_rows_file_name = f"processed_{segment_file}"
        selected_data.to_csv(os.path.join(output_dir, selected_rows_file_name), index=False)

        print(f"Selected rows saved for {segment_file}: {selected_rows_file_name}")


def run_posture_model(video_path):
    # Placeholder: 실제 모델을 로드하고 비디오를 분석하는 로직을 구현하세요.
    # 예시로 간단한 문자열을 반환합니다.
    get_frames(video_path)
    
    feedback_report = "video_path: " + video_path + "\n운동 자세가 전반적으로 양호합니다. 약간의 개선이 필요한 부분은 다음과 같습니다: 팔과 다리의 정렬을 더 신경 써주세요."
    return feedback_report