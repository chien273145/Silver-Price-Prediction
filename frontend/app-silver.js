/**
 * Silver Price Prediction Dashboard
 * Silver-Specific JavaScript Application
 * v2.2.0 - Separated from unified app.js
 */

// API Base URL
const API_BASE = window.location.origin;

// Fixed Asset Type - NO TOGGLE
const ASSET = 'silver';

// State
const state = {
    asset: ASSET,  // Fixed to silver
    currency: 'VND',
    historicalDays: 90,
    predictions: null,
    historical: null,
    realtime: null,
    modelInfo: null,
    news: null,
    marketAnalysis: null,
    chart: null,
    fearGreedChart: null,
    refreshInterval: null
};

// ========== PARTICLE EFFECTS ==========
function createParticles() {
    const container = document.getElementById('particles');
    if (!container) return;

    const particleCount = 20;

    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle silver';  // Always silver particles

        particle.style.left = `${Math.random() * 100}%`;
        particle.style.animationDelay = `${Math.random() * 15}s`;
        particle.style.animationDuration = `${15 + Math.random() * 10}s`;
        particle.style.width = `${3 + Math.random() * 4}px`;
        particle.style.height = particle.style.width;

        container.appendChild(particle);
    }
}

// ========== SMOOTH NUMBER ANIMATION ==========
function animateNumber(element, targetValue, duration = 800) {
    if (!element) return;

    const startValue = parseInt(element.textContent.replace(/[^\d.-]/g, '')) || 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        const easeProgress = 1 - Math.pow(1 - progress, 3);
        const currentValue = startValue + (targetValue - startValue) * easeProgress;

        if (state.currency === 'VND') {
            element.textContent = new Intl.NumberFormat('vi-VN').format(Math.round(currentValue));
        } else {
            element.textContent = new Intl.NumberFormat('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            }).format(currentValue);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ========== FEAR & GREED GAUGE CHART ==========
function createFearGreedGauge() {
    const ctx = document.getElementById('sentimentGauge');
    if (!ctx) return;

    if (state.fearGreedChart) {
        state.fearGreedChart.destroy();
    }

    state.fearGreedChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [0, 100],
                backgroundColor: [
                    '#e74c3c',
                    '#ecf0f1'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            circumference: 180,
            rotation: 270,
            cutout: '75%',
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            animation: {
                animateRotate: true,
                animateScale: false,
                duration: 1000
            }
        }
    });
}

function updateFearGreedGauge(score, signal, color) {
    if (!state.fearGreedChart) return;

    state.fearGreedChart.data.datasets[0].data = [score, 100 - score];
    state.fearGreedChart.data.datasets[0].backgroundColor[0] = color;
    state.fearGreedChart.update();

    const gaugeValue = document.getElementById('sentimentValue');
    const sentimentText = document.getElementById('sentimentText');

    if (gaugeValue) {
        gaugeValue.textContent = score;
        gaugeValue.style.color = color;
    }

    if (sentimentText) {
        sentimentText.textContent = signal;
        sentimentText.style.color = color;
    }
}

async function fetchFearGreedIndex() {
    try {
        const response = await fetch(`${API_BASE}/api/fear-greed`);
        const data = await response.json();

        if (data.success) {
            const { score, signal, color } = data.index;
            updateFearGreedGauge(score, signal, color);
        }
    } catch (error) {
        console.error('Error fetching Fear & Greed Index:', error);
        updateFearGreedGauge(50, 'NEUTRAL', '#f1c40f');
    }
}

// ========== MARKET NEWS ==========
// ========== MARKET NEWS ==========
// Note: News fetching is handled by loadNews() function later in the file


// ========== AI MARKET ANALYSIS ==========
async function fetchMarketAnalysis() {
    try {
        const container = document.getElementById('marketAnalysisContent');
        if (container) container.innerHTML = '<div class="analysis-loading">Đang phân tích thị trường bằng AI...</div>';

        const response = await fetch(`${API_BASE}/api/market-analysis?asset=${ASSET}`);
        const data = await response.json();

        if (data.success) {
            state.marketAnalysis = data;
            updateMarketAnalysisDisplay(data);
        } else {
            console.error('Failed to fetch market analysis:', data.error);
            if (container) container.innerHTML = '<div class="analysis-error">Không thể phân tích thị trường lúc này.</div>';
        }
    } catch (error) {
        console.error('Error fetching market analysis:', error);
    }
}

function updateMarketAnalysisDisplay(data) {
    const container = document.getElementById('marketAnalysisContent');
    if (!container) return;

    let html = `
        <div class="ai-analysis-header">
            <div class="ai-recommendation ${data.recommendation.action}">
                ${data.recommendation.text}
            </div>
        </div>
        <div class="analysis-grid">
    `;

    // Indicators
    if (data.indicators) {
        html += '<div class="analysis-indicators">';
        if (data.indicators.vix) {
            const vix = data.indicators.vix;
            html += `
                <div class="indicator-item ${vix.impact}">
                    <span class="ind-label">Chỉ số Sợ hãi (VIX):</span>
                    <span class="ind-value">${vix.value} (${vix.status})</span>
                </div>
            `;
        }
        if (data.indicators.dxy) {
            const dxy = data.indicators.dxy;
            html += `
                <div class="indicator-item ${dxy.impact}">
                    <span class="ind-label">Chỉ số USD (DXY):</span>
                    <span class="ind-value">${dxy.value} (${dxy.status})</span>
                </div>
            `;
        }
        if (data.indicators.gold_silver_ratio) {
            const gsr = data.indicators.gold_silver_ratio;
            html += `
                <div class="indicator-item">
                    <span class="ind-label">Tỷ lệ Vàng/Bạc:</span>
                    <span class="ind-value">${gsr.value} (${gsr.status})</span>
                </div>
            `;
        }
        html += '</div>';
    }

    // Analysis Points
    if (data.analysis && data.analysis.length > 0) {
        html += '<ul class="analysis-points">';
        data.analysis.forEach(point => {
            html += `<li>${point}</li>`;
        });
        html += '</ul>';
    }

    html += '</div>';
    container.innerHTML = html;
}

// ========== PERFORMANCE TRANSPARENCY ==========
async function fetchPerformanceTransparency() {
    try {
        // Silver uses the standard performance endpoint
        const response = await fetch(`${API_BASE}/api/performance-transparency`);
        const data = await response.json();

        if (data.success) {
            updatePerformanceDisplay(data.performance);
        }
    } catch (error) {
        console.error('Error fetching performance transparency:', error);
    }
}

function updatePerformanceDisplay(performance) {
    const accuracyContent = document.getElementById('accuracyContent');
    const accuracyBadge = document.getElementById('accuracyBadge');

    if (!accuracyContent) return;

    const { date, forecast, actual, difference, accuracy, model_confidence } = performance;

    const performanceHTML = `
        <div class="performance-grid">
            <div class="perf-item">
                <div class="perf-label">📅 Ngày</div>
                <div class="perf-value">${date}</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">🎯 Dự Báo</div>
                <div class="perf-value">${state.currency === 'VND' ?
            new Intl.NumberFormat('vi-VN').format(forecast.vnd) + ' VND' :
            '$' + forecast.usd}</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">📊 Thực Tế</div>
                <div class="perf-value">${state.currency === 'VND' ?
            new Intl.NumberFormat('vi-VN').format(actual.vnd) + ' VND' :
            '$' + actual.usd}</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">📈 Chênh Lệch</div>
                <div class="perf-value ${difference.percentage > 0 ? 'positive' : 'negative'}">
                    ${difference.percentage > 0 ? '+' : ''}${difference.percentage}%
                </div>
            </div>
            <div class="perf-item highlight">
                <div class="perf-label">✅ Độ Chính Xác</div>
                <div class="perf-value" style="color: ${accuracy.grade_color}">
                    ${accuracy.overall}% (${accuracy.grade})
                </div>
            </div>
            <div class="perf-item">
                <div class="perf-label">💪 Độ Tin Cậy</div>
                <div class="perf-value">${model_confidence === 'High' ? 'Cao' : 'Trung bình'}</div>
            </div>
        </div>
        <div class="performance-comment">
            <strong>${accuracy.comment}</strong>
        </div>
    `;

    accuracyContent.innerHTML = performanceHTML;

    if (accuracyBadge) {
        accuracyBadge.textContent = `${accuracy.overall}%`;
        accuracyBadge.style.backgroundColor = accuracy.grade_color;
    }
}

// ========== EXPLAINABLE AI ==========
async function fetchMarketFactors() {
    try {
        // Silver uses standard predict endpoint
        const response = await fetch(`${API_BASE}/api/predict`);
        const data = await response.json();

        if (data.success && data.market_drivers) {
            updateMarketFactorsDisplay(data.market_drivers);
        }
    } catch (error) {
        console.error('Error fetching market factors:', error);
    }
}

function updateMarketFactorsDisplay(marketDrivers) {
    const marketFactorsEl = document.getElementById('marketFactors');
    if (!marketFactorsEl) return;

    // VALIDITY CHECK: If no data or empty factors, KEEP STATIC CONTENT
    if (!marketDrivers ||
        (!marketDrivers.ai_explanation && (!marketDrivers.factors || marketDrivers.factors.length === 0))) {
        console.warn('No market factors data available available. Keeping static content.');
        return;
    }

    const aiExplanation = marketDrivers.ai_explanation;
    let factorsHTML = '';

    if (aiExplanation) {
        factorsHTML = `
            <div class="ai-summary">
                <div class="ai-summary-text">
                    <strong>🤖 Phân tích AI:</strong> ${aiExplanation.summary}
                </div>
                <div class="ai-confidence">
                    Mức độ tin cậy: <span class="confidence-${aiExplanation.confidence.toLowerCase()}">${aiExplanation.confidence}</span>
                </div>
            </div>
        `;

        if (aiExplanation.key_factors && aiExplanation.key_factors.length > 0) {
            factorsHTML += '<div class="dynamic-factors">';
            aiExplanation.key_factors.forEach(factor => {
                const impactClass = factor.impact === 'Tích cực' ? 'positive' : 'negative';
                const impactIcon = factor.impact === 'Tích cực' ? '📈' : '📉';

                factorsHTML += `
                    <div class="factor-item dynamic ${impactClass}">
                        <h4>
                            ${impactIcon} ${factor.factor}
                            <span class="factor-value">${factor.value}</span>
                        </h4>
                        <p class="factor-reason">${factor.reason}</p>
                        <div class="factor-impact ${impactClass}">
                            ${factor.impact}
                        </div>
                    </div>
                `;
            });
            factorsHTML += '</div>';
        }

        if (aiExplanation.outlook) {
            factorsHTML += `
                <div class="ai-outlook">
                    <h4>🔮 Dự báo thị trường</h4>
                    <p>${aiExplanation.outlook}</p>
                </div>
            `;
        }
    } else if (marketDrivers.factors && marketDrivers.factors.length > 0) {
        factorsHTML = marketDrivers.factors.map(factor =>
            `<div class="factor-item dynamic">
                <h4>📊 Phân tích thị trường</h4>
                <p>${factor}</p>
            </div>`
        ).join('');
    } else {
        // Double check fallthrough - should be caught by top check but just in case
        return;
    }

    marketFactorsEl.innerHTML = factorsHTML;
}

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
    metricR2: document.getElementById('metricR2'),
    metricMAPE: document.getElementById('metricMAPE'),
    newsList: document.getElementById('newsList'),
    refreshNewsBtn: document.getElementById('refreshNewsBtn'),
    accuracyBadge: document.getElementById('accuracyBadge'),
    accuracyContent: document.getElementById('accuracyContent'),
    sentimentValue: document.getElementById('sentimentValue'),
    sentimentText: document.getElementById('sentimentText'),
    marketFactors: document.getElementById('marketFactors'),
    calcBuyPrice: document.getElementById('calcBuyPrice'),
    calcAmount: document.getElementById('calcAmount'),
    calcCurrentPrice: document.getElementById('calcCurrentPrice'),
    calcPredPrice: document.getElementById('calcPredPrice'),
    calcProfitNow: document.getElementById('calcProfitNow'),
    calcProfitPred: document.getElementById('calcProfitPred')
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
    if (elements.loadingOverlay) {
        elements.loadingOverlay.classList.toggle('active', show);
    }
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

// API Functions - SILVER SPECIFIC ENDPOINTS
async function fetchAPI(endpoint) {
    try {
        const url = `${API_BASE}${endpoint}${endpoint.includes('?') ? '&' : '?'}t=${Date.now()}`;
        console.log(`[Silver] Fetching: ${url}`);
        const response = await fetch(url);
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
    // Always use silver endpoint
    const endpoint = `/api/predict?currency=${state.currency}`;
    console.log(`[Silver] getPredictions endpoint=${endpoint}`);
    return await fetchAPI(endpoint);
}

async function getHistorical() {
    // Always use silver endpoint
    const endpoint = `/api/historical?days=${state.historicalDays}&currency=${state.currency}`;
    return await fetchAPI(endpoint);
}

async function getRealtime() {
    return await fetchAPI('/api/realtime');
}

async function getModelInfo() {
    // Always use silver endpoint
    return await fetchAPI('/api/metrics');
}

async function getDataStatus() {
    return await fetchAPI('/api/data-status');
}

function updateDataStatus(status) {
    if (!status || !status.success || !elements.dataStatus) return;

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
    if (!state.predictions) {
        elements.currentPrice.textContent = '--';
        elements.priceUnit.textContent = '';
        if (elements.priceChange) {
            elements.priceChange.querySelector('.change-value').textContent = '--';
            elements.priceChange.querySelector('.change-percent').textContent = '--';
        }
        elements.predictedPrice.textContent = '--';
        if (elements.predictionChange) {
            elements.predictionChange.querySelector('.change-value').textContent = '--';
            elements.predictionChange.querySelector('.change-percent').textContent = '--';
        }
        elements.minPrice.textContent = '--';
        elements.maxPrice.textContent = '--';
        elements.avgPrice.textContent = '--';
        elements.exchangeRate.textContent = '--';
        return;
    }

    const { last_known, predictions, summary, exchange_rate, unit } = state.predictions;

    // Current price with animation (if not already set by local prices)
    if (!window.latestLocalPrices || !window.latestLocalPrices.items || window.latestLocalPrices.items.length === 0) {
        animateNumber(elements.currentPrice, last_known.price);
        elements.priceUnit.textContent = unit;
    }

    // Price change
    if (state.realtime && state.realtime.silver_price) {
        const silver = state.realtime.silver_price;
        if (silver.change !== null) {
            const changeValue = elements.priceChange.querySelector('.change-value');
            const changePercent = elements.priceChange.querySelector('.change-percent');

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
    if (!state.predictions) {
        elements.predictionTable.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 2rem;">Đang tải dữ liệu...</td></tr>';
        return;
    }

    const { predictions } = state.predictions;

    let html = '';
    predictions.forEach((pred) => {
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
    const modelR2 = document.getElementById('modelR2');
    const modelMAPE = document.getElementById('modelMAPE');
    const modelFeatures = document.getElementById('modelFeatures');
    const modelData = document.getElementById('modelData');

    if (!state.modelInfo || !state.modelInfo.model_info) {
        if (modelR2) modelR2.textContent = '--';
        if (modelMAPE) modelMAPE.textContent = '--';
        return;
    }

    const info = state.modelInfo.model_info;

    if (modelR2) modelR2.textContent = info.test_metrics?.r2?.toFixed(4) || info.r2 || '--';
    if (modelMAPE) modelMAPE.textContent = info.test_metrics?.mape ? `${info.test_metrics.mape.toFixed(2)}%` : (info.mape || '--');
    if (modelFeatures) modelFeatures.textContent = info.n_features || info.features || '--';
    if (modelData) modelData.textContent = info.data_sources || 'VIX, DXY, Oil';
}

function updateChart() {
    if (!state.historical || !state.predictions) return;

    const ctx = document.getElementById('priceChart').getContext('2d');

    // Filter data based on state.historicalDays
    const allData = state.historical.data;
    const filteredData = allData.slice(-state.historicalDays);

    const historicalData = filteredData.map(item => ({
        x: new Date(item.date),
        y: item.price
    }));

    const lastDate = new Date(state.predictions.last_known.date);
    const lastPrice = state.predictions.last_known.price;

    const predictionData = [
        { x: lastDate, y: lastPrice },
        ...state.predictions.predictions.map(item => ({
            x: new Date(item.date),
            y: item.price
        }))
    ];

    // Prepare confidence data (New)
    const confidenceLower = [
        { x: lastDate, y: lastPrice },
        ...state.predictions.predictions.map(item => ({
            x: new Date(item.date),
            y: item.lower !== undefined ? item.lower : item.price
        }))
    ];

    const confidenceUpper = [
        { x: lastDate, y: lastPrice },
        ...state.predictions.predictions.map(item => ({
            x: new Date(item.date),
            y: item.upper !== undefined ? item.upper : item.price
        }))
    ];

    if (state.chart) {
        state.chart.destroy();
    }

    // Silver-specific chart colors
    state.chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Giá lịch sử Bạc',
                    data: historicalData,
                    borderColor: 'rgba(192, 192, 192, 0.8)',
                    backgroundColor: 'rgba(192, 192, 192, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#c0c0c0',
                    borderWidth: 2,
                    spanGaps: true,  // Connect points across gaps
                    order: 3
                },
                {
                    label: 'Vùng tin cậy (Thấp)',
                    data: confidenceLower,
                    borderColor: 'transparent',
                    pointRadius: 0,
                    fill: false,
                    spanGaps: true,  // Connect points across gaps
                    order: 2
                },
                {
                    label: 'Vùng tin cậy (Cao)',
                    data: confidenceUpper,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(59, 130, 246, 0.15)', // Blue cloud to match Silver theme
                    pointRadius: 0,
                    fill: '-1', // Fill to previous dataset (Lower)
                    spanGaps: true,  // Connect points across gaps
                    order: 1
                },
                {
                    label: 'Dự đoán',
                    data: predictionData,
                    borderColor: 'rgba(59, 130, 246, 1)',
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    fill: false,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    spanGaps: true,  // Connect points across gaps
                    order: 0 // Top most
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            return `${context.dataset.label}: ${formatPrice(context.raw.y)}`;
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
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.5)'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.5)',
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
    console.log('🥈 Loading Silver data...');
    showLoading(true);

    try {
        const [predictionsData, historicalData, realtimeData, modelData, statusData] = await Promise.allSettled([
            getPredictions(),
            getHistorical(),
            getRealtime(),
            getModelInfo(),
            getDataStatus()
        ]);

        if (predictionsData.status === 'fulfilled' && predictionsData.value.success) {
            state.predictions = predictionsData.value;
            try { updatePriceCards(); } catch (e) { console.error('Error updating price cards:', e); }
            try { updatePredictionTable(); } catch (e) { console.error('Error updating prediction table:', e); }
        }

        if (historicalData.status === 'fulfilled' && historicalData.value.success) {
            state.historical = historicalData.value;
        }

        if (realtimeData.status === 'fulfilled' && realtimeData.value.success) {
            state.realtime = realtimeData.value;
            updatePriceCards();
        }

        if (modelData.status === 'fulfilled' && modelData.value.success) {
            state.modelInfo = modelData.value;
            updateModelMetrics();
        }

        if (statusData.status === 'fulfilled') {
            updateDataStatus(statusData.value);
        }

        updateChart();

        // Initial Fetch

        fetchFearGreedIndex();
        fetchPerformanceTransparency();
        fetchMarketFactors();
        loadNews();
        fetchMarketAnalysis();

        showToast('✅ Dữ liệu Bạc đã cập nhật', 'success');

    } catch (error) {
        console.error('Error loading data:', error);
        showToast('❌ Lỗi tải dữ liệu', 'error');
    } finally {
        showLoading(false);
    }
}

// Event Handlers (NO asset toggle needed)
function setupEventListeners() {
    // Period buttons (30, 90, 180 days)
    document.querySelectorAll('.period-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.historicalDays = parseInt(btn.dataset.days);
            getHistorical().then(data => {
                if (data.success) {
                    state.historical = data;
                    updateChart();
                }
            });
        });
    });

    // Currency buttons
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currency = btn.dataset.currency;
            loadData();
        });
    });

    // Refresh button
    if (elements.refreshBtn) {
        elements.refreshBtn.addEventListener('click', loadData);
    }

    // Auto refresh every 5 minutes
    if (state.refreshInterval) clearInterval(state.refreshInterval);
    state.refreshInterval = setInterval(() => {
        loadData();
        if (typeof fetchLocalPrices === 'function') {
            fetchLocalPrices();
        }
    }, 5 * 60 * 1000);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('🥈 Silver Price Prediction Dashboard');
    console.log('   Initializing...');

    createParticles();
    createFearGreedGauge();
    setupEventListeners();
    loadData();
    loadNews();
    loadAccuracy();

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

// ===== NEWS FUNCTIONS =====
async function loadNews() {
    if (!elements.newsList) return;

    try {
        const response = await fetch(`${API_BASE}/api/news?asset=silver`);
        const data = await response.json();

        if (data.success && data.articles && data.articles.length > 0) {
            displayNews(data.articles);
        } else {
            elements.newsList.innerHTML = '<div class="news-error">Không thể tải tin tức</div>';
        }
    } catch (error) {
        console.error('Error loading news:', error);
        elements.newsList.innerHTML = '<div class="news-error">Lỗi kết nối</div>';
    }
}

function displayNews(newsItems) {
    if (!elements.newsList) return;

    const html = newsItems.map(item => {
        let sentimentClass = '';
        const iconLower = (item.icon || '').toLowerCase();
        const titleLower = (item.title || '').toLowerCase();

        if (iconLower.includes('📈') || titleLower.includes('tăng')) {
            sentimentClass = 'bullish';
        } else if (iconLower.includes('📉') || titleLower.includes('giảm')) {
            sentimentClass = 'bearish';
        }

        return `
            <div class="news-item ${sentimentClass}">
                <span class="news-icon">${item.icon || '📰'}</span>
                <div class="news-title">${item.title}</div>
                <div class="news-source">${item.source} ${formatNewsDate(item.date)}</div>
            </div>
        `;
    }).join('');

    elements.newsList.innerHTML = html;
}

function formatNewsDate(dateStr) {
    if (!dateStr) return '';
    try {
        const date = new Date(dateStr);
        const now = new Date();
        const diffHours = Math.floor((now - date) / (1000 * 60 * 60));

        if (diffHours < 1) return 'Vừa xong';
        if (diffHours < 24) return `${diffHours} giờ trước`;
        if (diffHours < 48) return 'Hôm qua';
        return date.toLocaleDateString('vi-VN');
    } catch {
        return dateStr;
    }
}

// ===== ACCURACY FUNCTIONS =====
async function loadAccuracy() {
    if (!elements.accuracyContent) return;

    try {
        // Silver uses standard accuracy endpoint
        const response = await fetch(`${API_BASE}/api/accuracy?t=${Date.now()}`);
        const data = await response.json();

        if (data.success) {
            displayAccuracy(data);
        } else {
            elements.accuracyContent.innerHTML = '<p class="accuracy-loading">Chưa có dữ liệu</p>';
        }
    } catch (error) {
        console.error('Error loading accuracy:', error);
        elements.accuracyContent.innerHTML = '<p class="accuracy-loading">Lỗi kết nối</p>';
    }
}

function displayAccuracy(data) {
    if (!elements.accuracyContent || !elements.accuracyBadge) return;

    const acc = data.accuracy;

    // Determine grade and color
    let grade = 'A+';
    let gradeClass = 'positive';
    if (acc.overall >= 90) { grade = 'A+'; gradeClass = 'positive'; }
    else if (acc.overall >= 80) { grade = 'A'; gradeClass = 'positive'; }
    else if (acc.overall >= 70) { grade = 'B'; gradeClass = 'warning'; }
    else if (acc.overall >= 60) { grade = 'C'; gradeClass = 'warning'; }
    else { grade = 'D'; gradeClass = 'negative'; }

    elements.accuracyBadge.textContent = `${acc.overall}%`;
    elements.accuracyBadge.className = acc.overall >= 80 ? 'accuracy-badge' : 'accuracy-badge low';

    // Calculate direction accuracy class
    const dirClass = acc.direction >= 60 ? 'positive' : (acc.direction >= 50 ? '' : 'negative');
    const mapeClass = acc.mape <= 5 ? 'positive' : (acc.mape <= 10 ? 'warning' : 'negative');

    elements.accuracyContent.innerHTML = `
        <div class="accuracy-item">
            <div class="label"><span class="icon">🎯</span> Độ Chính Xác</div>
            <div class="value ${gradeClass}">${acc.overall}% <small style="font-size: 0.7em; opacity: 0.7">(${grade})</small></div>
        </div>
        <div class="accuracy-item">
            <div class="label"><span class="icon">📊</span> Xu Hướng</div>
            <div class="value ${dirClass}">${acc.direction}%</div>
        </div>
        <div class="accuracy-item">
            <div class="label"><span class="icon">📉</span> MAPE</div>
            <div class="value ${mapeClass}">${acc.mape}%</div>
        </div>
        <div class="accuracy-item">
            <div class="label"><span class="icon">💰</span> Sai Số TB</div>
            <div class="value">${acc.avg_error_usd ? '$' + acc.avg_error_usd : '--'}</div>
        </div>
        
        <div class="accuracy-summary ${acc.overall >= 80 ? '' : 'warning'}" style="grid-column: 1 / -1;">
            <span class="summary-icon">${acc.overall >= 80 ? '✅' : '⚠️'}</span>
            <span class="summary-text">
                ${acc.overall >= 80
            ? `<strong>Độ tin cậy cao</strong> - Mô hình dự đoán đang hoạt động tốt với độ chính xác ${acc.overall}%.`
            : `<strong>Độ tin cậy trung bình</strong> - Nên tham khảo thêm các nguồn khác khi ra quyết định.`
        }
            </span>
        </div>
    `;
}

// News refresh
if (elements.refreshNewsBtn) {
    elements.refreshNewsBtn.addEventListener('click', loadNews);
}

// ===== EMAIL SUBSCRIPTION =====
const subscribeForm = document.getElementById('subscribeForm');
if (subscribeForm) {
    subscribeForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const emailInput = document.getElementById('emailInput');
        const email = emailInput.value.trim();

        if (!email) return;

        try {
            const subscribers = JSON.parse(localStorage.getItem('subscribers') || '[]');

            if (subscribers.includes(email)) {
                showToast('Email đã được đăng ký trước đó!', 'warning');
                return;
            }

            subscribers.push(email);
            localStorage.setItem('subscribers', JSON.stringify(subscribers));

            emailInput.value = '';
            showToast('🎉 Đăng ký thành công! Cảm ơn bạn.', 'success');

            if (typeof gtag !== 'undefined') {
                gtag('event', 'subscribe', {
                    'event_category': 'engagement',
                    'event_label': 'email_subscription_silver'
                });
            }
        } catch (error) {
            console.error('Subscription error:', error);
            showToast('Có lỗi xảy ra. Vui lòng thử lại.', 'error');
        }
    });
}

// ===== CALCULATOR =====
function showCalculator() {
    const modal = document.getElementById('calculatorModal');
    if (modal) {
        modal.style.display = 'block';
        updateCalc();
    }
}

function updateCalc() {
    if (!elements.calcBuyPrice) return;

    const buyPrice = parseFloat(elements.calcBuyPrice.value) || 0;
    const amount = parseFloat(elements.calcAmount.value) || 1;

    let currentPriceVND = 0;
    let predPriceVND = 0;

    if (state.predictions && state.predictions.last_known) {
        currentPriceVND = state.predictions.last_known.price;
        predPriceVND = state.predictions.predictions[state.predictions.predictions.length - 1].price;
    }

    elements.calcCurrentPrice.textContent = currentPriceVND ? formatPrice(currentPriceVND) + ' VND' : '--';
    elements.calcPredPrice.textContent = predPriceVND ? formatPrice(predPriceVND) + ' VND' : '--';

    if (buyPrice > 0 && currentPriceVND > 0) {
        const spread = 0.02;
        const transactionFee = 0.001;
        const taxRate = 0.10;

        const sellPriceNow = currentPriceVND * (1 - spread / 2);
        const feeNow = sellPriceNow * transactionFee;
        const netSellNow = sellPriceNow - feeNow;
        const profitNow = (netSellNow - buyPrice) * amount;
        const profitNowAfterTax = profitNow > 0 ? profitNow * (1 - taxRate) : profitNow;
        const profitNowPercent = ((netSellNow - buyPrice) / buyPrice) * 100;

        elements.calcProfitNow.textContent = formatPrice(profitNowAfterTax) + ' VND';
        elements.calcProfitNow.className = profitNowAfterTax >= 0 ? 'calc-value positive' : 'calc-value negative';

        const profitNowPercentEl = document.getElementById('calcProfitNowPercent');
        if (profitNowPercentEl) {
            profitNowPercentEl.textContent = `${profitNowPercent >= 0 ? '+' : ''}${profitNowPercent.toFixed(2)}%`;
            profitNowPercentEl.className = profitNowPercent >= 0 ? 'calc-value positive' : 'calc-value negative';
        }

        const sellPricePred = predPriceVND * (1 - spread / 2);
        const feePred = sellPricePred * transactionFee;
        const netSellPred = sellPricePred - feePred;
        const profitPred = (netSellPred - buyPrice) * amount;
        const profitPredAfterTax = profitPred > 0 ? profitPred * (1 - taxRate) : profitPred;
        const profitPredPercent = ((netSellPred - buyPrice) / buyPrice) * 100;

        elements.calcProfitPred.textContent = formatPrice(profitPredAfterTax) + ' VND';
        elements.calcProfitPred.className = profitPredAfterTax >= 0 ? 'calc-value positive' : 'calc-value negative';

        const profitPredPercentEl = document.getElementById('calcProfitPredPercent');
        if (profitPredPercentEl) {
            profitPredPercentEl.textContent = `${profitPredPercent >= 0 ? '+' : ''}${profitPredPercent.toFixed(2)}%`;
            profitPredPercentEl.className = profitPredPercent >= 0 ? 'calc-value positive' : 'calc-value negative';
        }

        const bestCasePrice = predPriceVND * 1.10;
        const worstCasePrice = predPriceVND * 0.90;

        const bestCaseProfit = ((bestCasePrice * (1 - spread / 2) - buyPrice) * amount) * (profitPred > 0 ? (1 - taxRate) : 1);
        const worstCaseProfit = ((worstCasePrice * (1 - spread / 2) - buyPrice) * amount) * (profitPred > 0 ? (1 - taxRate) : 1);

        const bestCaseEl = document.getElementById('calcBestCase');
        const worstCaseEl = document.getElementById('calcWorstCase');

        if (bestCaseEl) {
            bestCaseEl.textContent = formatPrice(bestCaseProfit) + ' VND';
            bestCaseEl.className = bestCaseProfit >= 0 ? 'scenario-value positive' : 'scenario-value negative';
        }

        if (worstCaseEl) {
            worstCaseEl.textContent = formatPrice(worstCaseProfit) + ' VND';
            worstCaseEl.className = worstCaseProfit >= 0 ? 'scenario-value positive' : 'scenario-value negative';
        }
    }
}

if (elements.calcBuyPrice) {
    elements.calcBuyPrice.addEventListener('input', updateCalc);
    elements.calcAmount.addEventListener('input', updateCalc);
}

window.showCalculator = showCalculator;

// ========== LOCAL PRICES FETCHING (SILVER ONLY) ==========
async function fetchLocalPrices() {
    try {
        const tableBody = document.getElementById('localPriceTableBody');
        const timeBadge = document.getElementById('localPriceTime');
        if (!tableBody) return;

        const response = await fetch(`${API_BASE}/api/prices/local`);
        const result = await response.json();

        if (result.success && result.data.items.length > 0) {
            window.latestLocalPrices = result.data;
            if (window.updatePortfolioUI) window.updatePortfolioUI();

            tableBody.innerHTML = '';

            // Filter only SILVER products
            const filteredItems = result.data.items.filter(item => {
                const prodName = item.product_type.toUpperCase();
                // Exclude gold
                if (prodName.includes('VÀNG')) return false;
                // Must contain BẠC or SILVER
                return prodName.includes('BẠC') || prodName.includes('SILVER');
            });

            // Sort by spread
            const sortedItems = [...filteredItems].sort((a, b) => {
                const spreadA = a.sell_price - a.buy_price;
                const spreadB = b.sell_price - b.buy_price;
                return spreadA - spreadB;
            });

            const spreads = sortedItems.map(i => i.sell_price - i.buy_price);
            const minSpread = Math.min(...spreads);
            const maxSpread = Math.max(...spreads);

            sortedItems.forEach((item, index) => {
                const spread = item.sell_price - item.buy_price;
                const spreadPercent = item.sell_price > 0 ? ((spread / item.sell_price) * 100).toFixed(2) : 0;

                let spreadClass = 'spread-medium';
                if (spread === minSpread) spreadClass = 'spread-low';
                else if (spread === maxSpread || spreadPercent > 3) spreadClass = 'spread-high';

                const isBest = index === 0 ? 'best-spread-row' : '';

                const row = document.createElement('tr');
                row.className = isBest;
                row.innerHTML = `
                    <td><span class='brand-badge ${item.brand.toLowerCase()}'>${item.brand}</span></td>
                    <td>${item.product_type}</td>
                    <td class='price-up'>${new Intl.NumberFormat('vi-VN').format(item.buy_price)} ₫</td>
                    <td class='price-down'>${new Intl.NumberFormat('vi-VN').format(item.sell_price)} ₫</td>
                    <td class='${spreadClass}'>${new Intl.NumberFormat('vi-VN').format(spread)} ₫ <small>(${spreadPercent}%)</small></td>
                `;
                tableBody.appendChild(row);
            });

            // Update price card with best silver price
            if (filteredItems.length > 0) {
                let bestItem = filteredItems[0];
                const phuQuyMieng = filteredItems.find(i =>
                    i.product_type.includes('Bạc miếng') &&
                    i.product_type.includes('Phú Quý') &&
                    i.product_type.includes('1 lượng')
                );
                if (phuQuyMieng) {
                    bestItem = phuQuyMieng;
                } else {
                    const phuQuy = filteredItems.find(i => i.brand.match(/Phú Quý/i));
                    if (phuQuy) bestItem = phuQuy;
                }

                if (elements.currentPrice) {
                    animateNumber(elements.currentPrice, bestItem.sell_price);
                }
                if (elements.priceUnit) {
                    elements.priceUnit.textContent = "VND/Lượng (" + bestItem.brand + ")";
                }
            }

            if (timeBadge) timeBadge.textContent = 'Cập nhật: ' + new Date().toLocaleTimeString();
        } else {
            tableBody.innerHTML = '<tr><td colspan=5>Không thể tải dữ liệu</td></tr>';
        }
    } catch (e) {
        console.error('Error fetching local prices:', e);
    }
}

setTimeout(fetchLocalPrices, 1000);
setInterval(fetchLocalPrices, 300000);

// ========== AI BUY SCORE ==========
async function loadBuyScore() {
    try {
        const response = await fetch(`${API_BASE}/api/buy-score?asset=silver`);
        const data = await response.json();

        if (data.success) {
            displayBuyScore(data);
        }
    } catch (error) {
        console.error('Error loading buy score:', error);
    }
}

function displayBuyScore(scoreData) {
    const scoreEl = document.getElementById('buyScoreValue');
    const labelEl = document.getElementById('buyScoreLabel');
    const recommendationEl = document.getElementById('buyScoreRecommendation');
    const factorsEl = document.getElementById('buyScoreFactors');
    const assetEl = document.getElementById('buyScoreAsset');
    const card = document.querySelector('.buy-score-card');
    const ring = document.getElementById('buyScoreRing');

    if (!scoreEl) return;

    const score = scoreData.score;
    scoreEl.textContent = score;

    if (labelEl) labelEl.textContent = scoreData.label;
    if (recommendationEl) recommendationEl.textContent = scoreData.recommendation;

    if (assetEl) {
        assetEl.textContent = 'Bạc';
        assetEl.style.background = 'rgba(192, 192, 192, 0.2)';
        assetEl.style.color = 'var(--silver-primary)';
    }

    if (card) {
        if (score >= 60) card.setAttribute('data-score-level', 'high');
        else if (score >= 40) card.setAttribute('data-score-level', 'medium');
        else card.setAttribute('data-score-level', 'low');
    }

    if (ring) {
        const angle = -90 + (score / 100) * 180;
        ring.style.setProperty('--score-angle', angle);
    }

    if (factorsEl && scoreData.factors && scoreData.factors.length > 0) {
        const factorsHTML = scoreData.factors.map(factor => `
            <div class="factor-card">
                <div class="factor-header">
                    <span class="factor-name">${factor.icon} ${factor.name}</span>
                    <span class="factor-points">${factor.points}/${factor.max}</span>
                </div>
                <div class="factor-detail">${factor.detail}</div>
            </div>
        `).join('');
        factorsEl.innerHTML = factorsHTML;
    }
}

function toggleBuyScoreFactors() {
    const factorsEl = document.getElementById('buyScoreFactors');
    const toggleText = document.getElementById('buyScoreToggleText');

    if (factorsEl) {
        factorsEl.classList.toggle('expanded');
        if (toggleText) {
            toggleText.textContent = factorsEl.classList.contains('expanded') ? 'Ẩn chi tiết' : 'Xem chi tiết';
        }
    }
}

setTimeout(loadBuyScore, 2000);
setInterval(loadBuyScore, 300000);

// ========== MARKET NEWS ==========
async function loadNews() {
    const newsListEl = document.getElementById('newsList');
    if (!newsListEl) return;

    newsListEl.innerHTML = '<div class="news-loading">Đang tải tin tức...</div>';

    try {
        const response = await fetch(`${API_BASE}/api/news?tag=all`);
        const data = await response.json();

        if (!data.success || !data.news || data.news.length === 0) {
            newsListEl.innerHTML = '<div class="news-loading">Không có tin tức lúc này.</div>';
            return;
        }

        const sentimentColor = (score) => {
            if (score >= 60) return '#22c55e';
            if (score <= 40) return '#ef4444';
            return '#f59e0b';
        };

        const html = data.news.slice(0, 8).map(item => `
            <div class="news-item">
                <div class="news-header">
                    <span class="news-source">${item.source || ''}</span>
                    <span class="news-sentiment" style="color:${sentimentColor(item.sentiment_score || 50)};">
                        ${item.sentiment_label || 'Neutral'}
                    </span>
                </div>
                <a class="news-title" href="${item.link || '#'}" target="_blank" rel="noopener">
                    ${item.title || ''}
                </a>
                <div class="news-meta">
                    <span>${item.date ? new Date(item.date).toLocaleDateString('vi-VN') : ''}</span>
                </div>
            </div>
        `).join('');

        newsListEl.innerHTML = html;
    } catch (error) {
        console.error('Error loading news:', error);
        newsListEl.innerHTML = '<div class="news-loading">Không thể tải tin tức.</div>';
    }
}

// Attach refresh button
const refreshNewsBtn = document.getElementById('refreshNewsBtn');
if (refreshNewsBtn) refreshNewsBtn.addEventListener('click', loadNews);

setTimeout(loadNews, 3000);
setInterval(loadNews, 1800000); // Refresh every 30 min

console.log('🥈 app-silver.js v2.3.0 Loaded');
