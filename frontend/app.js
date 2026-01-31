/**
 * Silver Price Prediction Dashboard
 * Frontend JavaScript Application
 */

// API Base URL
const API_BASE = window.location.origin;

// State
const state = {
    currency: 'VND',
    historicalDays: 90,
    predictions: null,
    historical: null,
    realtime: null,
    modelInfo: null,
    chart: null,
    refreshInterval: null
};

// DOM Elements
const elements = {
    currentPrice: document.getElementById('currentPrice'),
    priceUnit: document.getElementById('priceUnit'),
    priceChange: document.getElementById('priceChange'),
    predictedPrice: document.getElementById('predictedPrice'),
    predictionChange: document.getElementById('predictionChange'),
    trendBadge: document.getElementById('trendBadge'),
    minPrice: document.getElementById('minPrice'),
    maxPrice: document.getElementById('maxPrice'),
    avgPrice: document.getElementById('avgPrice'),
    exchangeRate: document.getElementById('exchangeRate'),
    predictionTable: document.getElementById('predictionTable'),
    lastUpdate: document.getElementById('lastUpdate'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    toastContainer: document.getElementById('toastContainer'),
    refreshBtn: document.getElementById('refreshBtn'),
    dataStatus: document.getElementById('dataStatus'),
    metricRMSE: document.getElementById('metricRMSE'),
    metricMAE: document.getElementById('metricMAE'),
    metricR2: document.getElementById('metricR2'),
    metricMAPE: document.getElementById('metricMAPE')
};

// Utility Functions
function formatPrice(price, currency = state.currency) {
    if (price === null || price === undefined) return '--';

    if (currency === 'VND') {
        return new Intl.NumberFormat('vi-VN').format(Math.round(price));
    } else {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(price);
    }
}

function formatChange(change, isPercentage = false) {
    if (change === null || change === undefined) return '--';

    const sign = change >= 0 ? '+' : '';
    if (isPercentage) {
        return `${sign}${change.toFixed(2)}%`;
    }
    return `${sign}${formatPrice(change)}`;
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('vi-VN', {
        weekday: 'short',
        day: '2-digit',
        month: '2-digit'
    });
}

function showLoading(show = true) {
    elements.loadingOverlay.classList.toggle('active', show);
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span>${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span>
        <span>${message}</span>
    `;
    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// API Functions
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`);
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        throw error;
    }
}

async function getPredictions() {
    return await fetchAPI(`/api/predict?currency=${state.currency}`);
}

async function getHistorical() {
    return await fetchAPI(`/api/historical?days=${state.historicalDays}&currency=${state.currency}`);
}

async function getRealtime() {
    return await fetchAPI('/api/realtime');
}

async function getModelInfo() {
    return await fetchAPI('/api/metrics');
}

async function getDataStatus() {
    return await fetchAPI('/api/data-status');
}

// Update Data Status display
function updateDataStatus(status) {
    if (!status || !status.success) {
        elements.dataStatus.classList.remove('current', 'outdated');
        elements.dataStatus.querySelector('.status-text').textContent = 'Không có dữ liệu';
        return;
    }

    const lastDate = new Date(status.last_date);
    const formattedDate = lastDate.toLocaleDateString('vi-VN', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric'
    });

    if (status.is_current) {
        elements.dataStatus.classList.add('current');
        elements.dataStatus.classList.remove('outdated');
        elements.dataStatus.querySelector('.status-text').textContent = `Dữ liệu: ${formattedDate}`;
    } else {
        elements.dataStatus.classList.add('outdated');
        elements.dataStatus.classList.remove('current');
        elements.dataStatus.querySelector('.status-text').textContent = `Dữ liệu cũ: ${formattedDate} (${status.days_old} ngày)`;
    }
}

// Update Functions
function updatePriceCards() {
    if (!state.predictions) return;

    const { last_known, predictions, summary, exchange_rate, unit } = state.predictions;

    // Current price
    elements.currentPrice.textContent = formatPrice(last_known.price);
    elements.priceUnit.textContent = unit;

    // Price change (using realtime if available)
    if (state.realtime && state.realtime.silver_price) {
        const silver = state.realtime.silver_price;
        if (silver.change !== null) {
            const changeValue = elements.priceChange.querySelector('.change-value');
            const changePercent = elements.priceChange.querySelector('.change-percent');

            // Convert change to VND if needed
            let changeVND = silver.change;
            if (state.currency === 'VND' && exchange_rate) {
                changeVND = silver.change * 1.20565 * exchange_rate;
            }

            changeValue.textContent = formatChange(changeVND);
            changeValue.className = `change-value ${changeVND >= 0 ? 'positive' : 'negative'}`;

            const pct = silver.change_percent || 0;
            changePercent.textContent = formatChange(pct, true);
            changePercent.className = `change-percent ${pct >= 0 ? 'positive' : 'negative'}`;
        }
    }

    // Predicted price (day 7)
    const lastPrediction = predictions[predictions.length - 1];
    elements.predictedPrice.textContent = formatPrice(lastPrediction.price);

    // Prediction change
    const predChangeValue = elements.predictionChange.querySelector('.change-value');
    const predChangePercent = elements.predictionChange.querySelector('.change-percent');
    predChangeValue.textContent = formatChange(summary.total_change);
    predChangeValue.className = `change-value ${summary.total_change >= 0 ? 'positive' : 'negative'}`;
    predChangePercent.textContent = formatChange(summary.total_change_pct, true);
    predChangePercent.className = `change-percent ${summary.total_change_pct >= 0 ? 'positive' : 'negative'}`;

    // Trend badge
    elements.trendBadge.textContent = summary.trend === 'up' ? '📈 Tăng' : '📉 Giảm';
    elements.trendBadge.className = `trend-badge ${summary.trend}`;

    // Stats
    elements.minPrice.textContent = formatPrice(summary.min_price);
    elements.maxPrice.textContent = formatPrice(summary.max_price);
    elements.avgPrice.textContent = formatPrice(summary.avg_price);
    elements.exchangeRate.textContent = exchange_rate ?
        new Intl.NumberFormat('vi-VN').format(exchange_rate) + ' đ' : '--';
}

function updatePredictionTable() {
    if (!state.predictions) return;

    const { predictions, last_known } = state.predictions;

    let html = '';
    predictions.forEach((pred, index) => {
        const trendClass = pred.change.percentage >= 0 ? 'trend-up' : 'trend-down';
        const trendIcon = pred.change.percentage >= 0 ? '▲' : '▼';

        html += `
            <tr>
                <td>
                    <strong>Ngày ${pred.day}</strong><br>
                    <small style="color: var(--text-muted)">${formatDate(pred.date)}</small>
                </td>
                <td style="font-weight: 600">${formatPrice(pred.price)}</td>
                <td class="${trendClass}">${formatChange(pred.change.absolute)}</td>
                <td class="${trendClass}">${formatChange(pred.change.percentage, true)}</td>
                <td class="${trendClass}">${trendIcon}</td>
            </tr>
        `;
    });

    elements.predictionTable.innerHTML = html;
    elements.lastUpdate.textContent = `Cập nhật: ${new Date().toLocaleTimeString('vi-VN')}`;
}

function updateModelMetrics() {
    if (!state.modelInfo || !state.modelInfo.model_info) return;

    const info = state.modelInfo.model_info;

    if (info.test_metrics) {
        const metrics = info.test_metrics;
        elements.metricRMSE.textContent = `$${metrics.rmse?.toFixed(4) || '--'}`;
        elements.metricMAE.textContent = `$${metrics.mae?.toFixed(4) || '--'}`;
        elements.metricR2.textContent = metrics.r2?.toFixed(4) || '--';
        elements.metricMAPE.textContent = `${metrics.mape?.toFixed(2) || '--'}%`;
    }
}

function updateChart() {
    if (!state.historical || !state.predictions) return;

    const ctx = document.getElementById('priceChart').getContext('2d');

    // Prepare historical data
    const historicalData = state.historical.data.map(item => ({
        x: new Date(item.date),
        y: item.price
    }));

    // Prepare prediction data
    const lastDate = new Date(state.predictions.last_known.date);
    const lastPrice = state.predictions.last_known.price;

    const predictionData = [
        { x: lastDate, y: lastPrice },
        ...state.predictions.predictions.map(item => ({
            x: new Date(item.date),
            y: item.price
        }))
    ];

    // Destroy existing chart
    if (state.chart) {
        state.chart.destroy();
    }

    // Create new chart
    state.chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Giá lịch sử',
                    data: historicalData,
                    borderColor: 'rgba(192, 192, 192, 0.8)',
                    backgroundColor: 'rgba(192, 192, 192, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#c0c0c0',
                    borderWidth: 2
                },
                {
                    label: 'Dự đoán',
                    data: predictionData,
                    borderColor: 'rgba(59, 130, 246, 1)',
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 3,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#a0a0b0',
                        font: {
                            family: 'Inter',
                            size: 12
                        },
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(24, 24, 36, 0.95)',
                    titleColor: '#ffffff',
                    bodyColor: '#a0a0b0',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            const value = formatPrice(context.parsed.y);
                            const unit = state.currency === 'VND' ? 'VND' : 'USD';
                            return `${context.dataset.label}: ${value} ${unit}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'dd/MM'
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6a6a7a',
                        font: {
                            family: 'Inter',
                            size: 11
                        },
                        maxRotation: 0
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6a6a7a',
                        font: {
                            family: 'Inter',
                            size: 11
                        },
                        callback: function (value) {
                            return formatPrice(value);
                        }
                    }
                }
            }
        }
    });
}

// Load all data
async function loadData() {
    showLoading(true);
    elements.refreshBtn.classList.add('loading');

    try {
        // Fetch all data in parallel
        const [predictions, historical, realtime, modelInfo, dataStatus] = await Promise.all([
            getPredictions().catch(e => { console.error('Predictions error:', e); return null; }),
            getHistorical().catch(e => { console.error('Historical error:', e); return null; }),
            getRealtime().catch(e => { console.error('Realtime error:', e); return null; }),
            getModelInfo().catch(e => { console.error('Model info error:', e); return null; }),
            getDataStatus().catch(e => { console.error('Data status error:', e); return null; })
        ]);

        // Update state
        state.predictions = predictions;
        state.historical = historical;
        state.realtime = realtime;
        state.modelInfo = modelInfo;

        // Update UI
        if (predictions) {
            updatePriceCards();
            updatePredictionTable();
        }

        if (historical && predictions) {
            updateChart();
        }

        if (modelInfo) {
            updateModelMetrics();
        }

        if (dataStatus) {
            updateDataStatus(dataStatus);
        }

        showToast('Dữ liệu đã được cập nhật', 'success');

    } catch (error) {
        console.error('Error loading data:', error);
        showToast('Lỗi khi tải dữ liệu. Vui lòng thử lại.', 'error');
    } finally {
        showLoading(false);
        elements.refreshBtn.classList.remove('loading');
    }
}

// Event Handlers
function setupEventListeners() {
    // Currency toggle
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currency = btn.dataset.currency;
            loadData();
        });
    });

    // Period buttons
    document.querySelectorAll('.period-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.historicalDays = parseInt(btn.dataset.days);
            loadData();
        });
    });

    // Refresh button
    elements.refreshBtn.addEventListener('click', loadData);

    // Auto refresh every 5 minutes
    state.refreshInterval = setInterval(loadData, 5 * 60 * 1000);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('🥈 Silver Price Prediction Dashboard');
    console.log('   Initializing...');

    setupEventListeners();
    loadData();

    // Keyboard shortcut to refresh (R)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'r' && !e.ctrlKey && !e.metaKey) {
            loadData();
        }
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (state.refreshInterval) {
        clearInterval(state.refreshInterval);
    }
});
