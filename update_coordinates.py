import pandas as pd
import random

df = pd.read_csv('autos_solo.csv')

base_lat = -17.371995
base_lng = -66.162594

df['location_lat'] = df.apply(lambda x: base_lat + random.uniform(-0.0008, 0.0008), axis=1)
df['location_lng'] = df.apply(lambda x: base_lng + random.uniform(-0.0008, 0.0008), axis=1)

df.to_csv('autos_solo.csv', index=False)
print(f"Actualizado {len(df)} registros con coordenadas de Cochabamba")
print("\nPrimeros registros:")
print(df[['timestamp', 'object_class', 'location_lat', 'location_lng']].head(10))
