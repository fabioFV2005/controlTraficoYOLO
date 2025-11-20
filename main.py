import cv2
from ultralytics import YOLO
import numpy as np
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy.spatial import distance

def detect_objects_in_video(video_path, output_path=None, confidence_threshold=0.5):
    print("Cargando modelo YOLO...")
    model = YOLO('yolov8n.pt') 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"Resolución: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total de frames: {total_frames}")
    print(f"Duración: {total_frames/fps:.2f} segundos")
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    detections_summary = {}
    all_detections = []
    tracked_objects = {}
    next_object_id = 1
    max_distance_threshold = 100
    max_frames_disappeared = 30
    
    print("\nProcesando video...")
    
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
                object_id = matched_id
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
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{detection['class']}: {detection['confidence']:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - text_height - 10), 
                            (int(x1) + text_width, int(y1)), (0, 255, 0), -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        info_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        car_count = detections_summary.get('car', 0)
        counter_text = f"Autos: {car_count}"
        cv2.putText(frame, counter_text, (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if out:
            out.write(frame)
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progreso: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        cv2.imshow('Deteccion YOLO', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Procesamiento interrumpido por el usuario")
            break
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"Frames procesados: {frame_count}")
    
    if detections_summary:
        print("\n=== RESUMEN DE DETECCIONES ===")
        sorted_detections = sorted(detections_summary.items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_detections:
            print(f"{class_name}: {count} detecciones")
    else:
        print("No se detectaron objetos en el video.")
    
    if all_detections:
        csv_filename = "detecciones_completas.csv"
        cars_csv_filename = "autos_solo.csv"
        save_detections_to_csv(all_detections, csv_filename)
        show_csv_summary(all_detections)
        save_cars_only_csv(all_detections, cars_csv_filename)
        create_weekly_cars_chart(cars_csv_filename)
    else:
        print("No hay detecciones para guardar en CSV.")
    
    if output_path:
        print(f"\nVideo procesado guardado en: {output_path}")

def save_detections_to_csv(detections_data, csv_filename="detecciones.csv"):
    print(f"\nGuardando detecciones en CSV: {csv_filename}")
    fieldnames = [
        'timestamp',
        'frame_number', 
        'time_seconds',
        'object_class',
        'confidence',
        'bbox_x1',
        'bbox_y1', 
        'bbox_x2',
        'bbox_y2',
        'bbox_width',
        'bbox_height',
        'bbox_center_x',
        'bbox_center_y'
    ]
    
    try:
        # Verificar si el archivo existe para determinar si escribir encabezados
        file_exists = os.path.exists(csv_filename)
        
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Solo escribir encabezados si el archivo no existe o está vacío
            if not file_exists or os.path.getsize(csv_filename) == 0:
                writer.writeheader()
                print(f"  Archivo nuevo creado: {csv_filename}")
            else:
                print(f"  Agregando datos al archivo existente: {csv_filename}")
            
            for detection in detections_data:
                writer.writerow(detection)
        
        print(f"✓ CSV actualizado exitosamente con {len(detections_data)} nuevas detecciones")
        with open(csv_filename, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f) - 1
        print(f"  Total de detecciones en archivo: {total_lines}")
        
    except Exception as e:
        print(f"✗ Error al guardar CSV: {e}")

def show_csv_summary(detections_data):
    if not detections_data:
        return
    
    print(f"\n=== RESUMEN DEL ARCHIVO CSV ===")
    print(f"Total de detecciones guardadas: {len(detections_data)}")
    
    class_counts = {}
    confidence_sum = {}
    for detection in detections_data:
        obj_class = detection['object_class']
        confidence = detection['confidence']
        
        if obj_class not in class_counts:
            class_counts[obj_class] = 0
            confidence_sum[obj_class] = 0
        
        class_counts[obj_class] += 1
        confidence_sum[obj_class] += confidence
    
    print("\nDetecciones por clase:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        avg_confidence = confidence_sum[class_name] / count
        print(f"  {class_name}: {count} detecciones (confianza promedio: {avg_confidence:.3f})")
    
    first_frame = detections_data[0]['frame_number']
    last_frame = detections_data[-1]['frame_number']
    first_time = detections_data[0]['time_seconds']
    last_time = detections_data[-1]['time_seconds']
    
    print(f"\nRango temporal:")
    print(f"  Frames: {first_frame} - {last_frame}")
    print(f"  Tiempo: {first_time}s - {last_time}s")
    
    all_confidences = [d['confidence'] for d in detections_data]
    min_conf = min(all_confidences)
    max_conf = max(all_confidences)
    avg_conf = sum(all_confidences) / len(all_confidences)
    
    print(f"\nEstadísticas de confianza:")
    print(f"  Mínima: {min_conf:.3f}")
    print(f"  Máxima: {max_conf:.3f}")
    print(f"  Promedio: {avg_conf:.3f}")

def save_cars_only_csv(detections_data, csv_filename="autos_solo.csv"):
    car_detections = [detection for detection in detections_data if detection['object_class'] == 'car']
    
    if not car_detections:
        print("No se encontraron detecciones de autos para guardar.")
        return
    
    print(f"\nFiltrando y guardando solo autos en CSV: {csv_filename}")
    fieldnames = [
        'timestamp',
        'frame_number', 
        'time_seconds',
        'object_class',
        'confidence',
        'bbox_x1',
        'bbox_y1', 
        'bbox_x2',
        'bbox_y2',
        'bbox_width',
        'bbox_height',
        'bbox_center_x',
        'bbox_center_y'
    ]
    
    try:
        # Verificar si el archivo existe para determinar si escribir encabezados
        file_exists = os.path.exists(csv_filename)
        
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Solo escribir encabezados si el archivo no existe o está vacío
            if not file_exists or os.path.getsize(csv_filename) == 0:
                writer.writeheader()
                print(f"  Archivo nuevo de autos creado: {csv_filename}")
            else:
                print(f"  Agregando autos al archivo existente: {csv_filename}")
            
            for detection in car_detections:
                writer.writerow(detection)
        
        print(f"✓ CSV de autos actualizado exitosamente con {len(car_detections)} nuevas detecciones")
        
        with open(csv_filename, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f) - 1
        print(f"  Total de autos en archivo: {total_lines}")
        show_cars_summary(car_detections)
        
    except Exception as e:
        print(f"✗ Error al guardar CSV de autos: {e}")

def show_cars_summary(car_detections):
    if not car_detections:
        return
    
    print(f"\n=== RESUMEN DE DETECCIONES DE AUTOS ===")
    print(f"Total de autos detectados: {len(car_detections)}")
    
    confidences = [d['confidence'] for d in car_detections]
    min_conf = min(confidences)
    max_conf = max(confidences)
    avg_conf = sum(confidences) / len(confidences)
    
    print(f"\nEstadísticas de confianza (solo autos):")
    print(f"  Mínima: {min_conf:.3f}")
    print(f"  Máxima: {max_conf:.3f}")
    print(f"  Promedio: {avg_conf:.3f}")
    
    frames_with_cars = set(d['frame_number'] for d in car_detections)
    print(f"\nFrames con autos detectados: {len(frames_with_cars)}")
    avg_cars_per_frame = len(car_detections) / len(frames_with_cars)
    print(f"Promedio de autos por frame (con autos): {avg_cars_per_frame:.2f}")
    first_time = car_detections[0]['time_seconds']
    last_time = car_detections[-1]['time_seconds']
    print(f"Rango temporal de autos: {first_time}s - {last_time}s")
    frame_counts = {}
    for detection in car_detections:
        frame = detection['frame_number']
        frame_counts[frame] = frame_counts.get(frame, 0) + 1
    
    max_cars_frame = max(frame_counts.items(), key=lambda x: x[1])
    print(f"Frame con más autos: Frame {max_cars_frame[0]} ({max_cars_frame[1]} autos)")

def create_weekly_cars_chart(csv_filename="autos_solo.csv"):
    if not os.path.exists(csv_filename):
        print(f"No se encontró el archivo {csv_filename}")
        return
    
    print(f"\nGenerando gráfico de autos por día de la semana...")
    
    try:
        cars_by_day = defaultdict(int)
        with open(csv_filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                timestamp_str = row['timestamp']
                timestamp_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                day_of_week = timestamp_dt.weekday()
                cars_by_day[day_of_week] += 1
        days_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        days = []
        counts = []
        for i in range(7):
            days.append(days_names[i])
            counts.append(cars_by_day[i])
        plt.figure(figsize=(12, 8))
        bars = plt.bar(days, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'])
        plt.title('Cantidad de Autos Detectados por Día de la Semana', fontsize=16, fontweight='bold')
        plt.xlabel('Día de la Semana', fontsize=12)
        plt.ylabel('Cantidad de Autos', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        for bar, count in zip(bars, counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        str(count), ha='center', va='bottom', fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_filename = 'autos_por_dia_semana.png'
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Gráfico generado y guardado como: {chart_filename}")
        total_cars = sum(counts)
        print(f"\n=== ESTADÍSTICAS POR DÍA ===")
        for day, count in zip(days, counts):
            percentage = (count / total_cars * 100) if total_cars > 0 else 0
            print(f"{day}: {count} autos ({percentage:.1f}%)")
        print(f"\nTotal de autos analizados: {total_cars}")
        
    except Exception as e:
        print(f"✗ Error al generar gráfico: {e}")

def main():
    video_path = "1900-151662242_small.mp4"
    if not os.path.exists(video_path):
        print(f"Error: El archivo {video_path} no existe en el directorio actual")
        return
    output_path = "video_procesado_con_detecciones.mp4"
    
    detect_objects_in_video(
        video_path=video_path,
        output_path=output_path,
        confidence_threshold=0.5  
    )

if __name__ == "__main__":
    main()
