import time
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, Response
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

# Load dataset
new_df = pd.read_csv('Dataset/web_attacks_balanced_Random_Forest.csv')
new_df['Label'] = new_df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Shuffle the dataset
new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)

feature_columns = [
    'Average Packet Size',
    'Flow Bytes/s',
    'Max Packet Length',
    'Fwd Packet Length Mean',
    'Fwd IAT Min',
    'Total Length of Fwd Packets',
    'Flow IAT Mean',
    'Fwd IAT Std',
    'Fwd Packet Length Max',
    'Flow Packets/s'
]

# Remove rows with +infinity or -infinity in feature columns
new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
new_df.dropna(subset=feature_columns, inplace=True)

# Fit scaler on the feature columns
scaler = StandardScaler()
X_new = new_df[feature_columns].values
X_new_scaled = scaler.fit_transform(X_new)

y_new = new_df['Label'].values
y_new_cat = to_categorical(y_new, num_classes=2)

# Reshape for LSTM
X_new_lstm = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

# Load the saved LSTM model
loaded_lstm_model = load_model('Model/lstm_model.keras')

# Predict on the new data
y_pred_new_prob = loaded_lstm_model.predict(X_new_lstm)
y_pred_new = y_pred_new_prob.argmax(axis=1)

def generate_predictions():
    for idx, row in new_df.iterrows():
        features = row[feature_columns]
        pred_label = y_pred_new[idx]
        yield f"data: {{'features': {features.to_dict()}, 'prediction': '{pred_label}'}}\n\n"
        time.sleep(1)  # Simulate real-time (1 second per row)

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head>
            <title>Web Attack Detection Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                    font-family: 'Roboto', sans-serif;
                }
                body {
                    background: #f5f6fa;
                    padding: 20px;
                }
                .dashboard-header {
                    background: #fff;
                    padding: 30px 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    text-align: center;
                }
                .dashboard-title {
                    color: #2c3e50;
                    font-size: 32px;
                    font-weight: 500;
                    margin-bottom: 15px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                .stats-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .stat-card {
                    background: #fff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .stat-value {
                    font-size: 28px;
                    font-weight: 500;
                    margin-bottom: 5px;
                }
                .stat-label {
                    color: #7f8c8d;
                    font-size: 14px;
                }
                .charts-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .chart-card {
                    background: #fff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .chart-title {
                    color: #2c3e50;
                    font-size: 18px;
                    margin-bottom: 15px;
                }
                #results-table {
                    width: 100%;
                    border-collapse: collapse;
                    background: #fff;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    overflow: hidden;
                }
                #results-table th, #results-table td {
                    border: 1px solid #e9ecef;
                    padding: 12px 15px;
                }
                #results-table th {
                    background: #f8f9fa;
                    color: #2c3e50;
                    font-weight: 500;
                    text-align: left;
                }
                #results-table tbody {
                    display: block;
                    max-height: 400px;
                    overflow-y: auto;
                }
                #results-table thead, #results-table tbody tr {
                    display: table;
                    width: 100%;
                    table-layout: fixed;
                }
                .status-badge {
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 12px;
                    font-weight: 500;
                }
                .status-benign {
                    background: #e8f5e9;
                    color: #2e7d32;
                }
                .status-attack {
                    background: #ffebee;
                    color: #c62828;
                }
                .feature-data {
                    font-family: monospace;
                    font-size: 12px;
                    white-space: pre-wrap;
                    word-break: break-all;
                    background: #f8f9fa;
                    padding: 8px;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1 class="dashboard-title">Web Attack Detection Dashboard</h1>
                <p style="color: #7f8c8d; font-size: 16px; margin-top: 10px;">Real-time traffic analysis and threat detection</p>
            </div>
            
            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-value" id="totalRequests">0</div>
                    <div class="stat-label">Total Requests</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="benignCount">0</div>
                    <div class="stat-label">Benign Traffic</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="attackCount">0</div>
                    <div class="stat-label">Attack Traffic</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="attackRate">0%</div>
                    <div class="stat-label">Attack Rate</div>
                </div>
            </div>

            <div class="charts-container">
                <div class="chart-card">
                    <h3 class="chart-title">Traffic Distribution</h3>
                    <canvas id="barChart" width="400" height="300"></canvas>
                </div>
                <div class="chart-card">
                    <h3 class="chart-title">Traffic Analysis</h3>
                    <canvas id="pieChart" width="400" height="300"></canvas>
                </div>
            </div>

            <div class="chart-card">
                <h3 class="chart-title">Recent Traffic Analysis</h3>
                <table id="results-table">
                    <thead>
                        <tr>
                            <th style="width: 15%">Time</th>
                            <th style="width: 15%">Status</th>
                            <th style="width: 70%">Traffic Features</th>
                        </tr>
                    </thead>
                    <tbody id="results-body"></tbody>
                </table>
            </div>
            <script>
                var eventSource = new EventSource("/stream");
                var benignCount = 0, attackCount = 0;
                var timeLabels = [];
                var predData = [];
                var barChart, pieChart;
                var maxRows = 20;

                function updateCharts() {
                    // Bar chart update
                    barChart.data.datasets[0].data = [benignCount, attackCount];
                    barChart.update();
                    // Pie chart update with percentage
                    var total = benignCount + attackCount;
                    var benignPercent = total > 0 ? (benignCount / total * 100).toFixed(1) : 0;
                    var attackPercent = total > 0 ? (attackCount / total * 100).toFixed(1) : 0;
                    pieChart.data.datasets[0].data = [benignCount, attackCount];
                    pieChart.options.plugins = {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    var label = context.label || '';
                                    var value = context.parsed;
                                    var percent = label === 'Benign' ? benignPercent : attackPercent;
                                    return `${label}: ${value} (${percent}%)`;
                                }
                            }
                        }
                    };
                    pieChart.update();
                }

                window.onload = function() {
                    var ctxBar = document.getElementById('barChart').getContext('2d');
                    barChart = new Chart(ctxBar, {
                        type: 'bar',
                        data: {
                            labels: ['Benign', 'Attack'],
                            datasets: [{
                                label: 'Prediction Count',
                                data: [0, 0],
                                backgroundColor: ['#4caf50', '#f44336']
                            }]
                        },
                        options: {
                            responsive: false,
                            maintainAspectRatio: false,
                            scales: {y: {beginAtZero: true}},
                            plugins: {legend: {display: false}}
                        }
                    });
                    var ctxPie = document.getElementById('pieChart').getContext('2d');
                    pieChart = new Chart(ctxPie, {
                        type: 'pie',
                        data: {
                            labels: ['Benign', 'Attack'],
                            datasets: [{
                                data: [0, 0],
                                backgroundColor: ['#4caf50', '#f44336']
                            }]
                        },
                        options: {
                            responsive: false,
                            maintainAspectRatio: false,
                            plugins: {
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            var label = context.label || '';
                                            var value = context.parsed;
                                            var total = benignCount + attackCount;
                                            var percent = total > 0 ? (value / total * 100).toFixed(1) : 0;
                                            return `${label}: ${value} (${percent}%)`;
                                        }
                                    }
                                }
                            }
                        }
                    });
                };

                eventSource.onmessage = function(e) {
                    var data = JSON.parse(e.data.replace(/'/g, '"'));
                    var pred = parseInt(data.prediction);
                    if (pred === 0) benignCount++; else attackCount++;
                    
                    // Update stats
                    var total = benignCount + attackCount;
                    document.getElementById('totalRequests').textContent = total;
                    document.getElementById('benignCount').textContent = benignCount;
                    document.getElementById('attackCount').textContent = attackCount;
                    document.getElementById('attackRate').textContent = ((attackCount / total) * 100).toFixed(1) + '%';
                    
                    updateCharts();
                    
                    // Table update
                    var tbody = document.getElementById("results-body");
                    var row = document.createElement("tr");
                    var now = new Date().toLocaleTimeString();
                    var statusClass = pred === 0 ? 'status-benign' : 'status-attack';
                    var statusText = pred === 0 ? 'Benign' : 'Attack';
                    
                    row.innerHTML = `
                        <td>${now}</td>
                        <td><span class="status-badge ${statusClass}">${statusText}</span></td>
                        <td><div class="feature-data">${JSON.stringify(data.features, null, 2)}</div></td>
                    `;
                    tbody.insertBefore(row, tbody.firstChild);
                    while (tbody.rows.length > maxRows) tbody.deleteRow(-1);
                };
            </script>
        </body>
        </html>
    ''')

@app.route('/stream')
def stream():
    return Response(generate_predictions(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5050)
