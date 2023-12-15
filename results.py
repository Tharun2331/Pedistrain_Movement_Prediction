import cv2
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-output_path', help='Path for saved outputs')
target_vid = 'clip_000012.mp4'
target_filename = 'RIGA'
# Load CSV file
for detector in ['constant_velocity']:
    constant_velocity = pd.read_csv("outputs/constant_velocity/test_mask-rcnn_fold_1.csv")
    filtered_constant_velocity = constant_velocity[
    (constant_velocity['vid'] == target_vid) & 
    (constant_velocity['filename'] == target_filename)]

for detector in ['ground_truth']:
    ground_truth = pd.read_csv("outputs/ground_truth/test_mask-rcnn_fold_1.csv")
    filtered_ground_truth = ground_truth[
    (ground_truth['vid'] == target_vid) & 
    (ground_truth['filename'] == target_filename)]

for detector in ['sted']:
    predictions = pd.read_csv("outputs/sted/test_mask-rcnn_fold_1.csv")
    filtered_predictions = predictions[
    (predictions['vid'] == target_vid) & 
    (predictions['filename'] == target_filename)]


# Assuming the CSV file has columns like 'frame_num', 'x1', 'y1', 'x2', 'y2' for bounding box coordinates
video_path = './clips/RIGA/clip_000012.mp4'

video = cv2.VideoCapture(video_path)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Get the current frame number
    current_frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    # Get the predictions for the current frame
    frame_predictions = filtered_predictions[filtered_predictions['frame_num'] == current_frame_number]
    gt = filtered_ground_truth[filtered_ground_truth['frame_num'] == current_frame_number]
    cv = filtered_constant_velocity[filtered_constant_velocity['frame_num'] == current_frame_number]

    # Draw bounding boxes on the frame based on the predictions
    for index, row in frame_predictions.iterrows():
        x1, y1, x2, y2 = int(row['x1_1']), int(row['y1_1']), int(row['x2_1']), int(row['y2_1'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        x1, y1, x2, y2 = int(row['x1_60']), int(row['y1_60']), int(row['x2_60']), int(row['y2_60'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    for index,row in gt.iterrows():
        x1, y1, x2, y2 = int(row['x1_1']), int(row['y1_1']), int(row['x2_1']), int(row['y2_1'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        x1, y1, x2, y2 = int(row['x1_60']), int(row['y1_60']), int(row['x2_60']), int(row['y2_60'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for index,row in cv.iterrows():
        x1, y1, x2, y2 = int(row['x1_1']), int(row['y1_1']), int(row['x2_1']), int(row['y2_1'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        x1, y1, x2, y2 = int(row['x1_60']), int(row['y1_60']), int(row['x2_60']), int(row['y2_60'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


    # Display the frame with predictions
    cv2.imshow('Video with Predictions', frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

video.release()
cv2.destroyAllWindows()
