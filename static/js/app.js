mapboxgl.accessToken = 'pk.eyJ1IjoiZmFiaW9mZXJuYW5kZXp2eGQiLCJhIjoiY21hd3pmN281MGt6OTJtb2l4eTQ1emVmaCJ9.cNzeKYxiNAkYsXrJk1Offg';

let selectedLocation = null;
let currentVideoFile = null;
let weekChart = null;
let hourChart = null;

const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/light-v11',
    center: [-66.1568, -17.3935],
    zoom: 13
});

const marker = new mapboxgl.Marker({
    draggable: true,
    color: '#2b6cb0'
})
.setLngLat([-66.1568, -17.3935])
.addTo(map);

map.on('load', () => {
    map.addSource('heatmap-data', {
        type: 'geojson',
        data: {
            type: 'FeatureCollection',
            features: []
        }
    });
    
    map.addLayer({
        id: 'heatmap-layer',
        type: 'heatmap',
        source: 'heatmap-data',
        maxzoom: 18,
        paint: {
            'heatmap-weight': [
                'interpolate',
                ['linear'],
                ['get', 'intensity'],
                0, 0,
                100, 0.5,
                500, 1
            ],
            'heatmap-intensity': [
                'interpolate',
                ['linear'],
                ['zoom'],
                10, 1,
                15, 2,
                18, 3
            ],
            'heatmap-color': [
                'interpolate',
                ['linear'],
                ['heatmap-density'],
                0, 'rgba(33, 102, 172, 0)',
                0.2, 'rgb(103, 169, 207)',
                0.4, 'rgb(209, 229, 240)',
                0.6, 'rgb(253, 219, 199)',
                0.8, 'rgb(239, 138, 98)',
                1, 'rgb(178, 24, 43)'
            ],
            'heatmap-radius': [
                'interpolate',
                ['linear'],
                ['zoom'],
                10, 15,
                15, 25,
                18, 35
            ],
            'heatmap-opacity': 0.75
        }
    });
    
    map.addLayer({
        id: 'points-layer',
        type: 'circle',
        source: 'heatmap-data',
        minzoom: 16,
        paint: {
            'circle-radius': [
                'interpolate',
                ['linear'],
                ['get', 'intensity'],
                0, 3,
                100, 6,
                500, 10
            ],
            'circle-color': '#3182ce',
            'circle-opacity': 0.6,
            'circle-stroke-width': 2,
            'circle-stroke-color': '#fff'
        }
    });
    
    loadStats();
});

function updateLocationDisplay(lngLat) {
    selectedLocation = lngLat;
    document.getElementById('coords').textContent = 
        `Lat: ${lngLat.lat.toFixed(6)}, Lng: ${lngLat.lng.toFixed(6)}`;
}

marker.on('dragend', () => {
    updateLocationDisplay(marker.getLngLat());
});

map.on('click', (e) => {
    marker.setLngLat(e.lngLat);
    updateLocationDisplay(e.lngLat);
});

updateLocationDisplay(marker.getLngLat());

const uploadArea = document.getElementById('uploadArea');
const videoInput = document.getElementById('videoInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const uploadStatus = document.getElementById('uploadStatus');

uploadArea.addEventListener('click', () => videoInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragging');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragging');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragging');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
        handleVideoFile(file);
    }
});

videoInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleVideoFile(file);
    }
});

function handleVideoFile(file) {
    currentVideoFile = file;
    uploadStatus.className = 'success';
    uploadStatus.textContent = `Video cargado: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    analyzeBtn.disabled = false;
}

analyzeBtn.addEventListener('click', async () => {
    if (!currentVideoFile) return;
    
    const formData = new FormData();
    formData.append('video', currentVideoFile);
    formData.append('location', JSON.stringify({
        lat: selectedLocation.lat,
        lng: selectedLocation.lng
    }));
    
    document.getElementById('loadingModal').classList.add('active');
    
    try {
        const uploadResponse = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const uploadResult = await uploadResponse.json();
        
        if (uploadResult.success) {
            const locationParam = encodeURIComponent(JSON.stringify({
                lat: selectedLocation.lat,
                lng: selectedLocation.lng
            }));
            const analyzeResponse = await fetch(`/api/analyze/${uploadResult.filename}?location=${locationParam}`);
            const analyzeResult = await analyzeResponse.json();
            
            if (analyzeResult.success) {
                uploadStatus.className = 'success';
                uploadStatus.textContent = `Análisis completado: ${analyzeResult.cars_detected} autos detectados`;
                await loadStats();
            } else {
                throw new Error(analyzeResult.error);
            }
        } else {
            throw new Error(uploadResult.error);
        }
    } catch (error) {
        uploadStatus.className = 'error';
        uploadStatus.textContent = `Error: ${error.message}`;
    } finally {
        document.getElementById('loadingModal').classList.remove('active');
    }
});

async function loadStats(startDate = null, endDate = null, dayFilter = null) {
    try {
        let url = '/api/stats';
        const params = new URLSearchParams();
        
        if (dayFilter && dayFilter !== 'all') {
            params.append('day', dayFilter);
        }
        
        if (startDate || endDate) {
            if (startDate) params.append('start', startDate);
            if (endDate) params.append('end', endDate);
            url = '/api/stats/filter';
        }
        
        if (params.toString()) {
            url += `?${params.toString()}`;
        }
        
        const response = await fetch(url);
        const data = await response.json();
        
        updateCharts(data);
        updateStats(data);
        
        const dayNames = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'];
        if (dayFilter && dayFilter !== 'all') {
            document.getElementById('mapStats').textContent = 
                `Mostrando ${data.filtered_count} vehículos detectados el día ${dayNames[dayFilter]}`;
        } else {
            document.getElementById('mapStats').textContent = 
                `Mostrando ${data.filtered_count || data.total_cars} vehículos en total`;
        }
    } catch (error) {
        console.error('Error cargando estadísticas:', error);
    }
}

function updateCharts(data) {
    const dayLabels = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'];
    const dayMapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    };
    
    const dayData = new Array(7).fill(0);
    Object.entries(data.by_day).forEach(([day, count]) => {
        const index = dayMapping[day];
        if (index !== undefined) {
            dayData[index] = count;
        }
    });
    
    if (weekChart) {
        weekChart.destroy();
    }
    
    const weekCtx = document.getElementById('weekChart').getContext('2d');
    weekChart = new Chart(weekCtx, {
        type: 'bar',
        data: {
            labels: dayLabels,
            datasets: [{
                label: 'Vehículos Detectados',
                data: dayData,
                backgroundColor: dayData.map((value, index) => {
                    const colors = ['#3182ce', '#38a169', '#dd6b20', '#d69e2e', '#2c5282', '#2f855a', '#c05621'];
                    return colors[index];
                }),
                borderRadius: 8,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.parsed.y} vehículos`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Cantidad de Vehículos',
                        color: '#4a5568',
                        font: {
                            size: 12,
                            weight: 600
                        }
                    },
                    grid: {
                        color: '#f0f0f0'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Día de la Semana',
                        color: '#4a5568',
                        font: {
                            size: 12,
                            weight: 600
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
    
    if (data.by_hour) {
        const hourLabels = Array.from({length: 24}, (_, i) => {
            if (i === 0) return '12am';
            if (i < 12) return `${i}am`;
            if (i === 12) return '12pm';
            return `${i-12}pm`;
        });
        const hourData = Array.from({length: 24}, (_, i) => data.by_hour[i] || 0);
        
        if (hourChart) {
            hourChart.destroy();
        }
        
        const viewType = document.getElementById('hourViewType')?.value || 'line';
        const hourCtx = document.getElementById('hourChart').getContext('2d');
        
        hourChart = new Chart(hourCtx, {
            type: viewType,
            data: {
                labels: hourLabels,
                datasets: [{
                    label: 'Tráfico por Hora',
                    data: hourData,
                    borderColor: '#3182ce',
                    backgroundColor: viewType === 'line' 
                        ? 'rgba(49, 130, 206, 0.1)' 
                        : hourData.map(value => {
                            const max = Math.max(...hourData);
                            if (value === max) return '#e53e3e';
                            if (value > max * 0.7) return '#dd6b20';
                            if (value > max * 0.4) return '#d69e2e';
                            return '#3182ce';
                        }),
                    fill: viewType === 'line',
                    tension: 0.4,
                    borderWidth: 3,
                    borderRadius: viewType === 'bar' ? 8 : 0,
                    pointRadius: viewType === 'line' ? 4 : 0,
                    pointHoverRadius: viewType === 'line' ? 6 : 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.parsed.y} vehículos`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Cantidad de Vehículos',
                            color: '#4a5568',
                            font: {
                                size: 12,
                                weight: 600
                            }
                        },
                        grid: {
                            color: '#f0f0f0'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Hora del Día',
                            color: '#4a5568',
                            font: {
                                size: 12,
                                weight: 600
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
}

function updateStats(data) {
    document.getElementById('totalCars').textContent = data.total_cars || 0;
    document.getElementById('avgConfidence').textContent = 
        `${((data.avg_confidence || 0) * 100).toFixed(1)}%`;
    
    const dayNames = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    };
    
    if (data.by_day) {
        const maxDay = Object.entries(data.by_day).reduce((a, b) => a[1] > b[1] ? a : b);
        document.getElementById('peakDay').textContent = dayNames[maxDay[0]] || '-';
    }
    
    if (data.by_hour) {
        const maxHour = Object.entries(data.by_hour).reduce((a, b) => a[1] > b[1] ? a : b);
        document.getElementById('peakHour').textContent = `${maxHour[0]}:00`;
    }
    
    if (data.heatmap_data && map.getSource('heatmap-data')) {
        map.getSource('heatmap-data').setData({
            type: 'FeatureCollection',
            features: data.heatmap_data
        });
    }
}

document.getElementById('dayFilter').addEventListener('change', (e) => {
    const dayValue = e.target.value;
    loadStats(null, null, dayValue);
});

document.getElementById('hourViewType').addEventListener('change', () => {
    loadStats(
        document.getElementById('startDate').value || null,
        document.getElementById('endDate').value || null,
        document.getElementById('dayFilter').value || null
    );
});

document.getElementById('filterBtn').addEventListener('click', () => {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    const dayFilter = document.getElementById('dayFilter').value;
    loadStats(startDate, endDate, dayFilter);
});

document.getElementById('resetBtn').addEventListener('click', () => {
    document.getElementById('startDate').value = '';
    document.getElementById('endDate').value = '';
    document.getElementById('dayFilter').value = 'all';
    loadStats();
});
