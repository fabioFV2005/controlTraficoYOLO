import csv
import random
from datetime import datetime, timedelta

target_counts = {
    0: 1800,
    1: 2100,
    2: 1950,
    3: 2300,
    4: 2450,
    5: 1600,
    6: 1400
}

days_data = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: []
}

with open('autos_solo.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
        day = timestamp.weekday()
        days_data[day].append(row)

balanced_data = []

for day in range(7):
    current_count = len(days_data[day])
    target_count = target_counts[day]
    print(f"Día {day}: {current_count} registros -> objetivo: {target_count}")
    
    if current_count < target_count:
        needed = target_count - current_count
        print(f"  Generando {needed} registros adicionales...")
        
        for i in range(needed):
            if days_data[day]:
                template = random.choice(days_data[day])
                
                new_row = template.copy()
                
                base_date = datetime.strptime(template['timestamp'], '%Y-%m-%d %H:%M:%S')
                days_to_add = random.randint(-30, 30) * 7
                new_date = base_date + timedelta(days=days_to_add)
                new_date = new_date.replace(
                    hour=random.randint(6, 22),
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59)
                )
                new_row['timestamp'] = new_date.strftime('%Y-%m-%d %H:%M:%S')
                
                new_row['frame_number'] = str(random.randint(1, 462))
                new_row['time_seconds'] = str(round(int(new_row['frame_number']) / 29, 2))
                
                conf_base = float(template['confidence'])
                new_row['confidence'] = str(round(conf_base + random.uniform(-0.1, 0.1), 3))
                new_row['confidence'] = str(max(0.5, min(0.95, float(new_row['confidence']))))
                
                x_offset = random.uniform(-50, 50)
                y_offset = random.uniform(-30, 30)
                
                new_row['bbox_x1'] = str(round(float(template['bbox_x1']) + x_offset, 1))
                new_row['bbox_y1'] = str(round(float(template['bbox_y1']) + y_offset, 1))
                new_row['bbox_x2'] = str(round(float(template['bbox_x2']) + x_offset, 1))
                new_row['bbox_y2'] = str(round(float(template['bbox_y2']) + y_offset, 1))
                
                width = float(new_row['bbox_x2']) - float(new_row['bbox_x1'])
                height = float(new_row['bbox_y2']) - float(new_row['bbox_y1'])
                center_x = float(new_row['bbox_x1']) + width / 2
                center_y = float(new_row['bbox_y1']) + height / 2
                
                new_row['bbox_width'] = str(round(width, 1))
                new_row['bbox_height'] = str(round(height, 1))
                new_row['bbox_center_x'] = str(round(center_x, 1))
                new_row['bbox_center_y'] = str(round(center_y, 1))
                
                days_data[day].append(new_row)
    elif current_count > target_count:
        print(f"  Reduciendo a {target_count} registros...")
        days_data[day] = random.sample(days_data[day], target_count)
    
    balanced_data.extend(days_data[day])

balanced_data.sort(key=lambda x: (datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), int(x['frame_number'])))

fieldnames = [
    'timestamp', 'frame_number', 'time_seconds', 'object_class', 'confidence',
    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 
    'bbox_width', 'bbox_height', 'bbox_center_x', 'bbox_center_y'
]

with open('autos_solo.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(balanced_data)

print(f"\nDataset balanceado con {len(balanced_data)} registros totales")
print(f"Promedio por día: {len(balanced_data) / 7:.0f} registros")
