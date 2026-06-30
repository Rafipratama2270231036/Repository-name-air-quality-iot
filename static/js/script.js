/**
 * Air Quality Monitoring & Prediction - Dashboard Script
 * Production-ready, real-time updates, Chart.js, Bootstrap
 */
(function() {
    'use strict';

    // ========================
    // Global Variables
    // ========================
    const API_BASE = ''; // relative path, same domain
    const MAX_CHART_POINTS = 20; // jumlah titik data yang ditampilkan di chart

    // Chart instances
    let pm25Chart, tempChart, humChart;

    // Chart data buffers
    const pm25Data = [];
    const tempData = [];
    const humData = [];
    const timeLabels = [];

    // Interval ID for real-time fetching
    let realtimeInterval = null;

    // ========================
    // DOM Elements
    // ========================
    const pm25ValueEl = document.getElementById('pm25Value');
    const tempValueEl = document.getElementById('tempValue');
    const humValueEl = document.getElementById('humValue');
    const lastUpdateEl = document.getElementById('lastUpdate');
    const lastUpdateTextEl = document.getElementById('lastUpdateText');
    const statusBadge = document.getElementById('statusBadge');
    const predictDaysEl = document.getElementById('predictDays');
    const predictBtn = document.getElementById('predictBtn');
    const predictionResultContainer = document.getElementById('predictionResultContainer');
    const predictionTableBody = document.getElementById('predictionTableBody');
    const historyTableBody = document.getElementById('historyTableBody');
    const historyLoading = document.getElementById('historyLoading');

    // ========================
    // Utility Functions
    // ========================
    function formatDateTime(isoString) {
        if (!isoString) return '--';
        try {
            const date = new Date(isoString);
            return date.toLocaleString('id-ID', {
                day: '2-digit',
                month: 'short',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        } catch (e) {
            return isoString;
        }
    }

    function formatNumber(value, decimals = 1) {
        if (value === null || value === undefined || isNaN(value)) return '--';
        return Number(value).toFixed(decimals);
    }

    // Update status badge
    function setOnlineStatus(online) {
        if (online) {
            statusBadge.className = 'badge bg-success me-3';
            statusBadge.innerHTML = '<i class="fas fa-circle me-1"></i>Online';
        } else {
            statusBadge.className = 'badge bg-danger me-3';
            statusBadge.innerHTML = '<i class="fas fa-circle me-1"></i>Offline';
        }
    }

    // ========================
    // Real-time Data Fetching
    // ========================
    async function fetchLatest() {
        try {
            const response = await fetch(`${API_BASE}/api/latest`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const result = await response.json();

            if (result.status === 'success' && result.data) {
                const data = result.data;
                // Update card values
                pm25ValueEl.textContent = formatNumber(data.pm25, 1);
                tempValueEl.textContent = formatNumber(data.temperature, 1);
                humValueEl.textContent = formatNumber(data.humidity, 1);

                // Update last update
                const now = new Date();
                lastUpdateEl.textContent = `Last update: ${now.toLocaleTimeString('id-ID')}`;
                lastUpdateTextEl.textContent = formatDateTime(now.toISOString());

                // Set online status
                setOnlineStatus(true);

                // Update charts
                updateChartData(data);
            } else {
                // No data yet
                setOnlineStatus(true);
            }
        } catch (error) {
            console.error('Error fetching latest sensor data:', error);
            setOnlineStatus(false);
        }
    }

    function updateChartData(data) {
        const timestamp = new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        // Tambahkan data ke buffer
        timeLabels.push(timestamp);
        pm25Data.push(data.pm25);
        tempData.push(data.temperature);
        humData.push(data.humidity);

        // Batasi jumlah titik
        if (timeLabels.length > MAX_CHART_POINTS) {
            timeLabels.shift();
            pm25Data.shift();
            tempData.shift();
            humData.shift();
        }

        // Update chart PM2.5
        if (pm25Chart) {
            pm25Chart.data.labels = [...timeLabels];
            pm25Chart.data.datasets[0].data = [...pm25Data];
            pm25Chart.update('none'); // animasi minimal untuk performa
        }
        // Update chart Temperature
        if (tempChart) {
            tempChart.data.labels = [...timeLabels];
            tempChart.data.datasets[0].data = [...tempData];
            tempChart.update('none');
        }
        // Update chart Humidity
        if (humChart) {
            humChart.data.labels = [...timeLabels];
            humChart.data.datasets[0].data = [...humData];
            humChart.update('none');
        }
    }

    // ========================
    // Chart Initialization
    // ========================
    function initCharts() {
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0 // disable animation for performance in real-time
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Waktu'
                    },
                    ticks: {
                        maxTicksLimit: 10,
                        maxRotation: 45,
                        minRotation: 0
                    }
                },
                y: {
                    display: true,
                    beginAtZero: false
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        };

        // PM2.5 Chart
        const pm25Ctx = document.getElementById('pm25Chart').getContext('2d');
        pm25Chart = new Chart(pm25Ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'PM2.5 (µg/m³)',
                    data: [],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    pointRadius: 2,
                    pointBackgroundColor: '#0d6efd',
                    fill: true,
                    tension: 0.2
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        title: {
                            display: true,
                            text: 'µg/m³'
                        }
                    }
                }
            }
        });

        // Temperature Chart
        const tempCtx = document.getElementById('tempChart').getContext('2d');
        tempChart = new Chart(tempCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Suhu (°C)',
                    data: [],
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    borderWidth: 2,
                    pointRadius: 2,
                    pointBackgroundColor: '#dc3545',
                    fill: true,
                    tension: 0.2
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        title: {
                            display: true,
                            text: '°C'
                        }
                    }
                }
            }
        });

        // Humidity Chart
        const humCtx = document.getElementById('humChart').getContext('2d');
        humChart = new Chart(humCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Kelembapan (%)',
                    data: [],
                    borderColor: '#0dcaf0',
                    backgroundColor: 'rgba(13, 202, 240, 0.1)',
                    borderWidth: 2,
                    pointRadius: 2,
                    pointBackgroundColor: '#0dcaf0',
                    fill: true,
                    tension: 0.2
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        title: {
                            display: true,
                            text: '%'
                        },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    // ========================
    // Prediction Functions
    // ========================
    async function loadPrediction() {
        const days = parseInt(predictDaysEl.value, 10);
        if (isNaN(days) || days < 1 || days > 7) {
            alert('Pilih jumlah hari prediksi antara 1 sampai 7.');
            return;
        }

        // Disable button, show loading
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Memproses...';

        try {
            const response = await fetch(`${API_BASE}/api/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ days: days })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (result.status === 'success' && result.data && result.data.length > 0) {
                renderPredictionTable(result.data);
                predictionResultContainer.style.display = 'block';
            } else {
                alert('Prediksi gagal: data tidak tersedia.');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            alert(`Gagal melakukan prediksi: ${error.message}`);
        } finally {
            // Re-enable button
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<i class="fas fa-play me-2"></i>Prediksi';
        }
    }

    function renderPredictionTable(predictions) {
        let html = '';
        predictions.forEach(pred => {
            html += `
                <tr>
                    <td>${formatDateTime(pred.target_date)}</td>
                    <td>${formatNumber(pred.pm25_prediction, 1)}</td>
                    <td>${formatNumber(pred.temperature_prediction, 1)}</td>
                    <td>${formatNumber(pred.humidity_prediction, 1)}</td>
                </tr>
            `;
        });
        predictionTableBody.innerHTML = html;
    }

    // ========================
    // History Functions
    // ========================
    async function loadHistory() {
        historyLoading.classList.remove('d-none');
        try {
            const response = await fetch(`${API_BASE}/api/history?limit=100&offset=0`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const result = await response.json();
            if (result.status === 'success') {
                renderHistoryTable(result.data);
            } else {
                historyTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">Tidak ada data riwayat.</td></tr>';
            }
        } catch (error) {
            console.error('Error loading history:', error);
            historyTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-danger">Gagal memuat data riwayat.</td></tr>';
        } finally {
            historyLoading.classList.add('d-none');
        }
    }

    function renderHistoryTable(data) {
        if (!data || data.length === 0) {
            historyTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">Belum ada data sensor.</td></tr>';
            return;
        }

        let html = '';
        data.forEach(item => {
            html += `
                <tr>
                    <td>${formatDateTime(item.created_at)}</td>
                    <td>${formatNumber(item.pm25, 1)}</td>
                    <td>${formatNumber(item.temperature, 1)}</td>
                    <td>${formatNumber(item.humidity, 1)}</td>
                </tr>
            `;
        });
        historyTableBody.innerHTML = html;
    }

    // ========================
    // Event Listeners
    // ========================
    predictBtn.addEventListener('click', loadPrediction);

    // ========================
    // Initialization
    // ========================
    function startRealtimeUpdates() {
        // Fetch immediately on start
        fetchLatest();
        // Set interval every 1 detik (1000ms)
        realtimeInterval = setInterval(fetchLatest, 1000);
    }

    function stopRealtimeUpdates() {
        if (realtimeInterval) {
            clearInterval(realtimeInterval);
            realtimeInterval = null;
        }
    }

    // Initialize everything when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        initCharts();
        loadHistory();
        startRealtimeUpdates();
    });

    // Optional: Cleanup if needed (not strictly required but good practice)
    window.addEventListener('beforeunload', function() {
        stopRealtimeUpdates();
    });

})();