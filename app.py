from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
from ultralytics import YOLO
import os
import csv
from datetime import datetime
import pandas as pd
from werkzeug.utils import secure_filename
from scipy.spatial import distance
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'}), 400
    
    file = request.files['video']
    location_data = request.form.get('location')
    
    if file.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        location = json.loads(location_data) if location_data else None
        
        return jsonify({
            'success': True,
            'filename': filename,
            'location': location
        })
    
    return jsonify({'error': 'Formato no válido'}), 400

@app.route('/api/analyze/<filename>')
def analyze_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    location = request.args.get('location')
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Video no encontrado'}), 404
    
    location_data = json.loads(location) if location else None
    results = process_video_yolo(filepath, location=location_data)
    return jsonify(results)

def process_video_yolo(video_path, confidence_threshold=0.5, location=None):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {'error': 'No se pudo abrir el video'}
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    detections_summary = {}
    all_detections = []
    tracked_objects = {}
    next_object_id = 1
    max_distance_threshold = 100
    max_frames_disappeared = 30
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        results = model(frame, conf=confidence_threshold, verbose=False)
        
        current_frame_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_center_x = x1 + bbox_width / 2
                    bbox_center_y = y1 + bbox_height / 2
                    
                    current_frame_detections.append({
                        'center': (bbox_center_x, bbox_center_y),
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'width': bbox_width,
                        'height': bbox_height
                    })
        
        for obj_id in list(tracked_objects.keys()):
            tracked_objects[obj_id]['frames_disappeared'] += 1
            if tracked_objects[obj_id]['frames_disappeared'] > max_frames_disappeared:
                del tracked_objects[obj_id]
        
        for detection in current_frame_detections:
            matched_id = None
            min_distance = float('inf')
            
            for obj_id, tracked_obj in tracked_objects.items():
                if tracked_obj['class'] == detection['class']:
                    dist = distance.euclidean(detection['center'], tracked_obj['last_center'])
                    if dist < min_distance and dist < max_distance_threshold:
                        min_distance = dist
                        matched_id = obj_id
            
            if matched_id is not None:
                tracked_objects[matched_id]['last_center'] = detection['center']
                tracked_objects[matched_id]['frames_disappeared'] = 0
                tracked_objects[matched_id]['last_frame'] = frame_count
            else:
                object_id = next_object_id
                tracked_objects[object_id] = {
                    'class': detection['class'],
                    'last_center': detection['center'],
                    'frames_disappeared': 0,
                    'first_frame': frame_count,
                    'last_frame': frame_count
                }
                next_object_id += 1
                
                if detection['class'] not in detections_summary:
                    detections_summary[detection['class']] = 0
                detections_summary[detection['class']] += 1
                
                time_seconds = frame_count / fps
                x1, y1, x2, y2 = detection['bbox']
                
                detection_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'frame_number': frame_count,
                    'time_seconds': round(time_seconds, 2),
                    'object_class': detection['class'],
                    'confidence': round(float(detection['confidence']), 3),
                    'bbox_x1': round(float(x1), 1),
                    'bbox_y1': round(float(y1), 1),
                    'bbox_x2': round(float(x2), 1),
                    'bbox_y2': round(float(y2), 1),
                    'bbox_width': round(float(detection['width']), 1),
                    'bbox_height': round(float(detection['height']), 1),
                    'bbox_center_x': round(float(detection['center'][0]), 1),
                    'bbox_center_y': round(float(detection['center'][1]), 1)
                }
                all_detections.append(detection_data)
    
    cap.release()
    
    csv_filename = 'detecciones_completas.csv'
    save_to_csv(all_detections, csv_filename, location)
    
    cars_only = [d for d in all_detections if d['object_class'] == 'car']
    if cars_only:
        save_to_csv(cars_only, 'autos_solo.csv', location)
    
    return {
        'success': True,
        'total_frames': total_frames,
        'detections': detections_summary,
        'total_detections': len(all_detections),
        'cars_detected': len(cars_only)
    }

def save_to_csv(detections_data, csv_filename, location=None):
    fieldnames = [
        'timestamp', 'frame_number', 'time_seconds', 'object_class', 'confidence',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
        'bbox_width', 'bbox_height', 'bbox_center_x', 'bbox_center_y',
        'location_lat', 'location_lng'
    ]
    
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(csv_filename) == 0:
            writer.writeheader()
        for detection in detections_data:
            if location:
                detection['location_lat'] = location['lat']
                detection['location_lng'] = location['lng']
            writer.writerow(detection)

@app.route('/api/stats')
def get_stats():
    day_filter = request.args.get('day')
    
    if not os.path.exists('autos_solo.csv'):
        return jsonify({'error': 'No hay datos disponibles'}), 404
    
    df = pd.read_csv('autos_solo.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour
    
    if day_filter and day_filter != 'all':
        df = df[df['day_of_week'] == int(day_filter)]
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df.groupby('day_name').size().reindex(days_order, fill_value=0).to_dict()
    
    hour_counts = df.groupby('hour').size().to_dict()
    
    confidence_avg = df['confidence'].mean()
    
    heatmap_features = []
    if 'location_lat' in df.columns and 'location_lng' in df.columns:
        location_groups = df.groupby(['location_lat', 'location_lng']).size().reset_index(name='count')
        for _, row in location_groups.iterrows():
            if pd.notna(row['location_lat']) and pd.notna(row['location_lng']):
                heatmap_features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [row['location_lng'], row['location_lat']]
                    },
                    'properties': {
                        'intensity': int(row['count'])
                    }
                })
    
    return jsonify({
        'by_day': day_counts,
        'by_hour': hour_counts,
        'total_cars': len(df),
        'avg_confidence': round(confidence_avg, 3),
        'heatmap_data': heatmap_features,
        'filtered_count': len(df)
    })

@app.route('/api/stats/filter')
def get_filtered_stats():
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    if not os.path.exists('autos_solo.csv'):
        return jsonify({'error': 'No hay datos disponibles'}), 404
    
    df = pd.read_csv('autos_solo.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if start_date:
        df = df[df['timestamp'] >= start_date]
    if end_date:
        df = df[df['timestamp'] <= end_date]
    
    df['day_name'] = df['timestamp'].dt.day_name()
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df.groupby('day_name').size().reindex(days_order, fill_value=0).to_dict()
    
    return jsonify({
        'by_day': day_counts,
        'total_cars': len(df)
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
