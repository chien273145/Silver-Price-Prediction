/**
 * Silver Price Prediction Dashboard
 * Frontend JavaScript Application
 */

// API Base URL
const API_BASE = window.location.origin;

// State
const state = {
    asset: document.body.dataset.asset || 'silver',  // 'silver' or 'gold'
    currency: 'VND',
    historicalDays: 90,
    predictions: null,
    historical: null,
    realtime: null,
    modelInfo: null,
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
        particle.className = `particle ${Math.random() > 0.5 ? 'gold' : 'silver'}`;

        // Random position and animation
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

        // Easing function (ease-out-cubic)
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

    // Destroy existing chart if it exists
    if (state.fearGreedChart) {
        state.fearGreedChart.destroy();
    }

    state.fearGreedChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [0, 100],
                backgroundColor: [
                    '#e74c3c',  // Red for fear
                    '#ecf0f1'   // Light gray for background
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

    // Update gauge data
    state.fearGreedChart.data.datasets[0].data = [score, 100 - score];
    state.fearGreedChart.data.datasets[0].backgroundColor[0] = color;
    state.fearGreedChart.update();

    // Update center text
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
            const { score, signal, color, components } = data.index;
            updateFearGreedGauge(score, signal, color);

            // Optional: Log components for debugging
            console.log('Fear & Greed Components:', components);
        }
    } catch (error) {
        console.error('Error fetching Fear & Greed Index:', error);
        // Fallback to neutral
        updateFearGreedGauge(50, 'NEUTRAL', '#f1c40f');
    }
}

// ========== PERFORMANCE TRANSPARENCY ==========
// ========== PERFORMANCE TRANSPARENCY ==========
function updatePerformanceDisplay() {
    // Use data embedded in predictions response
    if (!state.predictions || !state.predictions.accuracy_check) return;

    const accuracy = state.predictions.accuracy_check;
    const accuracyContent = document.getElementById('accuracyContent');
    const accuracyBadge = document.getElementById('accuracyBadge');

    if (!accuracyContent) return;

    const diffClass = accuracy.diff >= 0 ? 'positive' : 'negative';
    const diffSign = accuracy.diff >= 0 ? '+' : '';

    const performanceHTML = `
        <div class="performance-grid">
            <div class="perf-item">
                <div class="perf-label">📅 Ngày</div>
                <div class="perf-value">${formatDate(accuracy.date)}</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">🎯 Dự báo (Hôm qua)</div>
                <div class="perf-value">${formatPrice(accuracy.predicted)} <small>${accuracy.unit || ''}</small></div>
            </div>
            <div class="perf-item">
                <div class="perf-label">📊 Thực tế (Hôm nay)</div>
                <div class="perf-value">${formatPrice(accuracy.actual)} <small>${accuracy.unit || ''}</small></div>
            </div>
            <div class="perf-item">
                <div class="perf-label">📈 Chênh lệch</div>
                <div class="perf-value ${diffClass}">
                    ${diffSign}${formatPrice(accuracy.diff)} (${accuracy.diff_pct}%)
                </div>
            </div>
            <div class="perf-item highlight">
                <div class="perf-label">✅ Độ chính xác</div>
                <div class="perf-value" style="color: #00d97e">
                    ${accuracy.accuracy}%
                </div>
            </div>
        </div>
        <div class="performance-comment">
            <strong>AI đã dự đoán chính xác ${accuracy.accuracy}% hôm nay.</strong>
        </div>
    `;

    accuracyContent.innerHTML = performanceHTML;

    if (accuracyBadge) {
        accuracyBadge.textContent = `${accuracy.accuracy}%`;
        accuracyBadge.style.backgroundColor = accuracy.accuracy >= 95 ? '#00d97e' : (accuracy.accuracy >= 90 ? '#f1c40f' : '#e74c3c');
    }
}

// ========== EXPLAINABLE AI ==========
async function fetchMarketFactors() {
    try {
        const endpoint = state.asset === 'gold' ?
            `${API_BASE}/api/gold/predict` :
            `${API_BASE}/api/predict`;

        const response = await fetch(endpoint);
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

    // Get AI explanation if available
    const aiExplanation = marketDrivers.ai_explanation;

    // Build dynamic factors HTML
    let factorsHTML = '';

    if (aiExplanation) {
        // AI-Generated Explanation
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

        // Key factors with detailed explanations
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

        // AI Outlook
        if (aiExplanation.outlook) {
            factorsHTML += `
                <div class="ai-outlook">
                    <h4>🔮 Dự báo thị trường</h4>
                    <p>${aiExplanation.outlook}</p>
                </div>
            `;
        }
    } else {
        // Fallback to static factors
        factorsHTML = marketDrivers.factors.map(factor =>
            `<div class="factor-item dynamic">
                <h4>📊 Phân tích thị trường</h4>
                <p>${factor}</p>
            </div>`
        ).join('');
    }

    marketFactorsEl.innerHTML = factorsHTML;
}

// ========== UPDATE UI FOR ASSET TYPE ==========
function updateUIForAsset() {
    const isGold = state.asset === 'gold';

    // Update logo and titles
    if (elements.logoIcon) elements.logoIcon.textContent = isGold ? '🥇' : '🥈';
    if (elements.appTitle) elements.appTitle.textContent = isGold ? 'Gold Price AI' : 'Silver Price AI';
    if (elements.appSubtitle) elements.appSubtitle.textContent = isGold ? 'Dự đoán giá vàng thông minh' : 'Dự đoán giá bạc thông minh';

    // Update model badge
    const modelBadge = document.getElementById('modelBadge');
    if (modelBadge) {
        modelBadge.textContent = isGold ? 'Gold Ridge + VIX/DXY' : 'Silver Ridge Regression';
    }

    // Update model metrics for gold
    if (isGold) {
        const modelR2 = document.getElementById('modelR2');
        const modelMAPE = document.getElementById('modelMAPE');
        const modelFeatures = document.getElementById('modelFeatures');
        const modelData = document.getElementById('modelData');

        if (modelR2) modelR2.textContent = '0.97';
        if (modelMAPE) modelMAPE.textContent = '3.44%';
        if (modelFeatures) modelFeatures.textContent = '91';
        if (modelData) modelData.textContent = 'VIX, DXY, Oil';
    } else {
        const modelR2 = document.getElementById('modelR2');
        const modelMAPE = document.getElementById('modelMAPE');
        const modelFeatures = document.getElementById('modelFeatures');
        const modelData = document.getElementById('modelData');

        if (modelR2) modelR2.textContent = '0.96';
        if (modelMAPE) modelMAPE.textContent = '3.37%';
        if (modelFeatures) modelFeatures.textContent = '44';
        if (modelData) modelData.textContent = 'GS Ratio, GPR';
    }

    // Update document title
    document.title = isGold
        ? 'Gold Price Prediction | Premium AI Forecasting'
        : 'Silver Price Prediction | Premium AI Forecasting';

    // Update Local Price Table Header
    const localPriceHeader = document.getElementById('localPriceHeader');
    if (localPriceHeader) {
        localPriceHeader.textContent = isGold
            ? '🇻🇳 Bảng Giá Vàng Trong Nước (SJC, DOJI, PNJ)'
            : '🇻🇳 Bảng Giá Bạc (SJC, Thế Giới)';
    }
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
    metricRMSE: document.getElementById('metricRMSE'),
    metricMAE: document.getElementById('metricMAE'),
    metricR2: document.getElementById('metricR2'),
    metricMAPE: document.getElementById('metricMAPE'),
    // New elements
    newsList: document.getElementById('newsList'),
    refreshNewsBtn: document.getElementById('refreshNewsBtn'),
    accuracyBadge: document.getElementById('accuracyBadge'),
    accuracyBadge: document.getElementById('accuracyBadge'),
    accuracyContent: document.getElementById('accuracyContent'),
    // New Elements
    sentimentValue: document.getElementById('sentimentValue'),
    sentimentText: document.getElementById('sentimentText'),
    marketFactors: document.getElementById('marketFactors'),
    // Calculator
    calcBuyPrice: document.getElementById('calcBuyPrice'),
    calcAmount: document.getElementById('calcAmount'),
    calcCurrentPrice: document.getElementById('calcCurrentPrice'),
    calcPredPrice: document.getElementById('calcPredPrice'),
    calcProfitNow: document.getElementById('calcProfitNow'),
    calcProfitPred: document.getElementById('calcProfitPred'),
    calcAsset: document.getElementById('calcAsset'),
    // Asset toggle elements
    logoIcon: document.getElementById('logoIcon'),
    appTitle: document.getElementById('appTitle'),
    appSubtitle: document.getElementById('appSubtitle'),
    // New Elements
    sentimentValue: document.getElementById('sentimentValue'),
    sentimentText: document.getElementById('sentimentText'),
    marketFactors: document.getElementById('marketFactors'),
    // Calculator
    calcBuyPrice: document.getElementById('calcBuyPrice'),
    calcAmount: document.getElementById('calcAmount'),
    calcCurrentPrice: document.getElementById('calcCurrentPrice'),
    calcPredPrice: document.getElementById('calcPredPrice'),
    calcProfitNow: document.getElementById('calcProfitNow'),
    calcProfitPred: document.getElementById('calcProfitPred'),
    calcAsset: document.getElementById('calcAsset')
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
        // Add timestamp to prevent caching
        const url = `${API_BASE}${endpoint}${endpoint.includes('?') ? '&' : '?'}t=${Date.now()}`;
        console.log(`Fetching: ${url}`);
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
    const endpoint = state.asset === 'gold'
        ? `/api/gold/predict?currency=${state.currency}`
        : `/api/predict?currency=${state.currency}`;

    // DEBUG: Log endpoint choice
    console.log(`[DEBUG] getPredictions asset=${state.asset} endpoint=${endpoint}`);

    const data = await fetchAPI(endpoint);
    updateDebugInfo(endpoint, data);
    return data;
}

async function getHistorical() {
    const endpoint = state.asset === 'gold'
        ? `/api/gold/historical?days=${state.historicalDays}&currency=${state.currency}`
        : `/api/historical?days=${state.historicalDays}&currency=${state.currency}`;
    return await fetchAPI(endpoint);
}

async function getRealtime() {
    return await fetchAPI('/api/realtime');
}

async function getModelInfo() {
    const endpoint = state.asset === 'gold' ? '/api/gold/metrics' : '/api/metrics';
    return await fetchAPI(endpoint);
}

async function getDataStatus() {
    return await fetchAPI('/api/data-status');
}

// Update Data Status display
// Update Data Status display (Debug Info Disabled)
function updateDebugInfo(endpoint, data) {
    // Debug info disabled for production UI
    /*
    let debugEl = document.getElementById('debug-info');
    if (!debugEl) {
        debugEl = document.createElement('div');
        debugEl.id = 'debug-info';
        debugEl.style.position = 'fixed';
        debugEl.style.bottom = '10px';
        debugEl.style.left = '10px';
        debugEl.style.background = 'rgba(0,0,0,0.85)';
        debugEl.style.color = '#00ff00';
        debugEl.style.padding = '12px';
        debugEl.style.borderRadius = '8px';
        debugEl.style.fontSize = '12px';
        debugEl.style.zIndex = '99999';
        debugEl.style.fontFamily = 'monospace';
        debugEl.style.pointerEvents = 'none';
        debugEl.style.border = '1px solid #333';
        document.body.appendChild(debugEl);
    }

    const price = data?.last_known?.price
        ? new Intl.NumberFormat('en-US').format(data.last_known.price)
        : 'N/A';

    const firstPred = data?.predictions && data.predictions[0]
        ? new Intl.NumberFormat('en-US').format(data.predictions[0].price)
        : 'N/A';

    debugEl.innerHTML = `
        <div style="margin-bottom:4px;color:#fff;font-weight:bold">🔍 DEBUG INFO</div>
        <div>Asset State: <span style="color:${state.asset === 'gold' ? 'gold' : 'silver'}">${state.asset.toUpperCase()}</span></div>
        <div>Endpoint: ${endpoint.split('?')[0]}</div>
        <div>Last Price: ${price}</div>
        <div>Pred[0]: ${firstPred}</div>
        <div>Success: ${data?.success}</div>
        <div>Time: ${new Date().toLocaleTimeString()}</div>
    `;
    */
}

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

    // Current price with animation
    // ONLY update from prediction data if local prices haven't already set a value
    // (Local prices from fetchLocalPrices() take priority - Phú Quý for Silver, SJC for Gold)
    if (!window.latestLocalPrices || !window.latestLocalPrices.items || window.latestLocalPrices.items.length === 0) {
        animateNumber(elements.currentPrice, last_known.price);
        elements.priceUnit.textContent = unit;
    }

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

    // Update range (New)
    const rangeEl = document.getElementById('predictedRange');
    if (rangeEl && lastPrediction.lower && lastPrediction.upper) {
        const low = formatPrice(lastPrediction.lower);
        const high = formatPrice(lastPrediction.upper);
        rangeEl.textContent = `Vùng an toàn: ${low} - ${high}`;
    } else if (rangeEl) {
        rangeEl.textContent = '';
    }

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

    // Update performance widget (transparency)
    updatePerformanceDisplay();
}

function updatePredictionTable() {
    if (!state.predictions) {
        elements.predictionTable.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 2rem;">Đang tải dữ liệu...</td></tr>';
        return;
    }

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
    if (!state.modelInfo || !state.modelInfo.model_info) {
        elements.metricRMSE.textContent = '--';
        elements.metricMAE.textContent = '--';
        elements.metricR2.textContent = '--';
        elements.metricMAPE.textContent = '--';
        return;
    }

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
                    borderWidth: 2,
                    order: 3
                },
                {
                    label: 'Vùng tin cậy (Thấp)',
                    data: confidenceLower,
                    borderColor: 'transparent',
                    pointRadius: 0,
                    fill: false,
                    order: 2
                },
                {
                    label: 'Vùng tin cậy (Cao)',
                    data: confidenceUpper,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(59, 130, 246, 0.15)', // Light blue cloud
                    pointRadius: 0,
                    fill: '-1', // Fill to previous dataset (Lower)
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
                    order: 0 // Top most
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
            },
            // Zoom plugin configuration
            plugins: {
                zoom: {
                    zoom: {
                        wheel: {
                            enabled: true,
                            modifierKey: 'ctrl'
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'xy',
                        onZoomComplete: function ({ chart }) {
                            chart.update('none');
                        }
                    },
                    pan: {
                        enabled: true,
                        mode: 'xy',
                    },
                    limits: {
                        y: { min: 0 }
                    }
                }
            }
        }
    });

    // Add zoom hint below chart
    const chartContainer = document.getElementById('priceChart').parentElement;
    if (!document.querySelector('.chart-zoom-hint')) {
        const hint = document.createElement('div');
        hint.className = 'chart-zoom-hint';
        hint.innerHTML = '💡 Ctrl + Scroll để zoom | Kéo để di chuyển | <button onclick="resetChartZoom()" style="color: var(--info); background: none; border: none; cursor: pointer; text-decoration: underline;">Reset</button>';
        chartContainer.appendChild(hint);
    }
}

// Reset chart zoom
function resetChartZoom() {
    if (state.chart) {
        state.chart.resetZoom();
    }
}

// Load all data
async function loadData() {
    showLoading(true);
    elements.refreshBtn.classList.add('loading');

    // Clear previous state to avoid mixing data
    const currentAsset = state.asset;
    console.log(`Loading data for: ${currentAsset} (${state.currency})`);

    try {
        // Fetch all data in parallel
        const [predictions, historical, realtime, modelInfo, dataStatus] = await Promise.all([
            getPredictions().catch(e => { console.error('Predictions error:', e); return null; }),
            getHistorical().catch(e => { console.error('Historical error:', e); return null; }),
            getRealtime().catch(e => { console.error('Realtime error:', e); return null; }),
            getModelInfo().catch(e => { console.error('Model info error:', e); return null; }),
            getDataStatus().catch(e => { console.error('Data status error:', e); return null; })
        ]);

        // Fetch additional features after main data
        const [fearGreed, performance, marketFactors] = await Promise.all([
            fetchFearGreedIndex().catch(e => { console.error('Fear & Greed error:', e); }),
            fetchPerformanceTransparency().catch(e => { console.error('Performance error:', e); }),
            fetchMarketFactors().catch(e => { console.error('Market factors error:', e); })
        ]);

        // Verify we are still on the same asset (race condition check)
        if (state.asset !== currentAsset) {
            console.log('Asset changed during load, ignoring result');
            return;
        }

        // Update state
        state.predictions = predictions;
        state.historical = historical;
        state.realtime = realtime;
        state.modelInfo = modelInfo;

        // Update UI
        let updated = false;

        if (predictions && predictions.success) {
            updatePriceCards();
            updatePredictionTable();
            updated = true;
        } else {
            console.error('No predictions data received or success=false', predictions);
            showToast('Không thể tải dữ liệu dự đoán', 'error');
            // Reset UI
            updatePriceCards();
            updatePredictionTable();
        }

        if (historical && historical.success && predictions && predictions.success) {
            updateChart();
        } else {
            if (state.chart) state.chart.destroy();
        }

        if (modelInfo) {
            updateModelMetrics();
        }

        if (dataStatus) {
            updateDataStatus(dataStatus);
        }

        // NEW UPDATES
        if (predictions && predictions.market_drivers) {
            updateMarketFactors(predictions.market_drivers);
            updateSentiment(predictions.summary, predictions.market_drivers);
        } else if (predictions && predictions.summary) {
            updateSentiment(predictions.summary, null);
        }

        if (predictions && predictions.accuracy_check) {
            updateAccuracy(predictions.accuracy_check);
        }

        // NEW UPDATES
        if (predictions && predictions.market_drivers) {
            updateMarketFactors(predictions.market_drivers);
            updateSentiment(predictions.summary, predictions.market_drivers);
        } else if (predictions && predictions.summary) {
            updateSentiment(predictions.summary, null);
        }

        if (predictions && predictions.accuracy_check) {
            updateAccuracy(predictions.accuracy_check);
        }

        if (updated) {
            showToast('Dữ liệu đã được cập nhật', 'success');
        }

    } catch (error) {
        console.error('Error loading data:', error);

        // Detailed error message for user
        let msg = 'Lỗi khi tải dữ liệu.';
        if (error.message.includes('503')) msg += ' Server đang khởi động mô hình (503).';
        else if (error.message.includes('500')) msg += ' Lỗi server (500).';
        else if (error.message.includes('Network')) msg += ' Lỗi kết nối mạng.';

        showToast(`${msg} Vui lòng thử lại.`, 'error');

        // Reset UI to empty state on error
        updatePriceCards();
        updatePredictionTable();
        updateModelMetrics();
        if (state.chart) state.chart.destroy();

    } finally {
        showLoading(false);
        elements.refreshBtn.classList.remove('loading');
    }
}

// Event Handlers
function setupEventListeners() {
    // Period buttons (30, 90, 180 days)
    document.querySelectorAll('.period-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            updateChart(parseInt(btn.dataset.days));
        });
    });



    // Currency buttons
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currency = btn.dataset.currency;

            // Refresh displays
            updateUIForAsset();
            if (state.historicalDays) updateChart(state.historicalDays);
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
    const assetEmoji = state.asset === 'gold' ? '🥇' : '🥈';
    console.log(`${assetEmoji} Metal Price Prediction Dashboard`);
    console.log('   Initializing...');

    createParticles();
    createFearGreedGauge();
    updateUIForAsset();
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
        const response = await fetch(`${API_BASE}/api/news`);
        const data = await response.json();

        if (data.success && data.news && data.news.length > 0) {
            displayNews(data.news);
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
        // Determine sentiment class for colored left border
        let sentimentClass = '';
        const iconLower = (item.icon || '').toLowerCase();
        const titleLower = (item.title || '').toLowerCase();

        // Detect sentiment from icon or title content
        if (iconLower.includes('📈') || iconLower.includes('🟢') || iconLower.includes('💹') ||
            titleLower.includes('tăng') || titleLower.includes('hỗ trợ') || titleLower.includes('tích cực')) {
            sentimentClass = 'bullish';
        } else if (iconLower.includes('📉') || iconLower.includes('🔴') ||
            titleLower.includes('giảm') || titleLower.includes('suy yếu') || titleLower.includes('tiêu cực')) {
            sentimentClass = 'bearish';
        } else if (iconLower.includes('🏛') || iconLower.includes('📊')) {
            sentimentClass = 'neutral';
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
        const endpoint = state.asset === 'gold' ? '/api/gold/accuracy' : '/api/accuracy';
        const response = await fetch(`${API_BASE}${endpoint}?t=${Date.now()}`);
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

    // Update badge
    elements.accuracyBadge.textContent = `${acc.overall}%`;
    elements.accuracyBadge.className = acc.overall >= 90 ? 'accuracy-badge' : 'accuracy-badge low';

    // Update content
    elements.accuracyContent.innerHTML = `
        <div class="accuracy-item">
            <div class="label">Độ chính xác tổng</div>
            <div class="value ${acc.overall >= 90 ? 'positive' : ''}">${acc.overall}%</div>
        </div>
        <div class="accuracy-item">
            <div class="label">Chính xác xu hướng</div>
            <div class="value ${acc.direction >= 60 ? 'positive' : ''}">${acc.direction}%</div>
        </div>
        <div class="accuracy-item">
            <div class="label">MAPE (Sai số %)</div>
            <div class="value">${acc.mape}%</div>
        </div>
        <div class="accuracy-item">
            <div class="label">Sai số trung bình</div>
            <div class="value">$${acc.avg_error_usd}</div>
        </div>
    `;
}

// Add event listener for news refresh
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
            // For now, just save to localStorage and show success
            // Later can be replaced with actual API call
            const subscribers = JSON.parse(localStorage.getItem('subscribers') || '[]');

            if (subscribers.includes(email)) {
                showToast('Email đã được đăng ký trước đó!', 'warning');
                return;
            }

            subscribers.push(email);
            localStorage.setItem('subscribers', JSON.stringify(subscribers));

            emailInput.value = '';
            showToast('🎉 Đăng ký thành công! Cảm ơn bạn.', 'success');

            // Track in Google Analytics if available
            if (typeof gtag !== 'undefined') {
                gtag('event', 'subscribe', {
                    'event_category': 'engagement',
                    'event_label': 'email_subscription'
                });
            }
        } catch (error) {
            console.error('Subscription error:', error);
            showToast('Có lỗi xảy ra. Vui lòng thử lại.', 'error');
        }
    });
}

// ===== HELPER FUNCTIONS =====
function showPrivacyPolicy() {
    alert('Chính sách bảo mật:\n\n' +
        '• Chúng tôi chỉ sử dụng email để gửi thông báo dự đoán giá.\n' +
        '• Không chia sẻ thông tin với bên thứ ba.\n' +
        '• Bạn có thể hủy đăng ký bất cứ lúc nào.\n' +
        '• Dữ liệu được lưu trữ an toàn.');
}

function showContact() {
    alert('Liên hệ:\n\n' +
        '📧 Email: support@silverprice.ai\n' +
        '🌐 Website: dubaovangbac.com');
}

// ===== GAUGE CHART & SENTIMENT =====
function initGauge() {
    const ctx = document.getElementById('sentimentGauge').getContext('2d');

    state.gaugeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Tiêu cực', 'Trung tính', 'Tích cực'],
            datasets: [{
                data: [33, 33, 33],
                backgroundColor: [
                    '#ef4444', // Red (Fear)
                    '#eab308', // Yellow (Neutral)
                    '#22c55e'  // Green (Greed)
                ],
                borderWidth: 0,
                circumference: 180,
                rotation: 270,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            }
        }
    });
}

function updateSentiment(summary, drivers) {
    if (!state.gaugeChart) initGauge();

    let score = 50; // Neutral default
    let text = "Chờ tín hiệu...";
    let color = "#eab308";

    // Logic based on Prediction Change
    const changePct = summary.total_change_pct;

    if (changePct > 0.5) {
        score = 75; // Greed / Buy
        text = "MUA MẠNH";
        color = "#22c55e";
    } else if (changePct < -0.5) {
        score = 25; // Fear / Sell
        text = "BÁN MẠNH";
        color = "#ef4444";
    } else {
        score = 50;
        text = "TRUNG LẬP";
    }

    // Adjust by RSI if available
    let suffix = "";
    if (drivers && drivers.raw && drivers.raw.rsi) {
        const rsi = drivers.raw.rsi.value;
        if (rsi > 70) {
            suffix = " (Quá Mua)";
            // If price predicted up but RSI overbought -> Caution
        } else if (rsi < 30) {
            suffix = " (Quá Bán)";
        }
    }

    // Update Value Display
    if (elements.sentimentValue) {
        elements.sentimentValue.textContent = text;
        elements.sentimentValue.style.color = color;
    }
    if (elements.sentimentText) {
        elements.sentimentText.textContent = `Dự báo xu hướng ${summary.trend === 'up' ? 'Tăng' : 'Giảm'} ${Math.abs(changePct).toFixed(2)}%${suffix}`;
    }

    // Highlight chart section
    const colors = ['#334155', '#334155', '#334155']; // Dimmed
    if (score > 60) colors[2] = '#22c55e';
    else if (score < 40) colors[0] = '#ef4444';
    else colors[1] = '#eab308';

    state.gaugeChart.data.datasets[0].backgroundColor = colors;
    state.gaugeChart.update();
}

// ===== MARKET FACTORS =====
function updateMarketFactors(drivers) {
    if (!drivers || !drivers.factors || !elements.marketFactors) return;

    let html = '';
    drivers.factors.forEach(factor => {
        html += `
            <div class="factor-item dynamic">
                <h4>🔔 Tín hiệu AI</h4>
                <p>${factor}</p>
            </div>
        `;
    });

    // Add default static if empty
    if (drivers.factors.length === 0) {
        html = '<p style="color:#a0a0b0; padding:10px; font-style:italic">Chưa có tín hiệu nổi bật hôm nay.</p>';
    }

    // Append standard educational factors below
    html += `
        <div style="margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px;">
             <small style="color: #64748b; display: block; margin-bottom: 5px;">KIẾN THỨC CƠ BẢN:</small>
             <div class="factor-item" style="opacity: 0.8">
                <h4>Lãi Suất & USD</h4>
                <p>Lãi suất FED tăng thường làm tăng giá trị USD, gây áp lực giảm lên giá vàng/bạc.</p>
            </div>
        </div>
    `;

    elements.marketFactors.innerHTML = html;
}

// ===== ACCURACY CHECK =====
function updateAccuracy(check) {
    if (!check || !elements.accuracyContent) return;

    if (elements.accuracyBadge) {
        elements.accuracyBadge.textContent = `${check.accuracy.toFixed(1)}%`;
        elements.accuracyBadge.className = `accuracy-badge ${check.accuracy > 95 ? 'high' : 'med'}`;
    }

    elements.accuracyContent.innerHTML = `
        <div class="accuracy-row">
            <span class="label">Dự báo hôm qua:</span>
            <span class="value">${formatPrice(check.predicted_usd, 'USD')}</span>
        </div>
        <div class="accuracy-row">
            <span class="label">Thực tế hôm nay:</span>
            <span class="value">${formatPrice(check.actual_usd, 'USD')}</span>
        </div>
        <div class="accuracy-row">
            <span class="label">Sai số:</span>
            <span class="value ${check.diff_usd >= 0.5 ? 'highlight' : ''}">${check.diff_usd.toFixed(2)} USD</span>
        </div>
        <div class="accuracy-note">*So sánh giá đóng cửa USD/oz</div>
    `;
}

// ===== NEWS SENTIMENT =====
async function fetchNews() {
    try {
        const response = await fetch(`${API_BASE}/api/news?asset=${state.asset}`);
        const data = await response.json();

        if (data.success) {
            updateNewsDisplay(data.news);
        } else {
            console.error('Failed to fetch news:', data.message);
        }
    } catch (error) {
        console.error('Error fetching news:', error);
    }
}

function updateNewsDisplay(newsData) {
    if (!elements.newsList) return;

    if (!newsData || newsData.length === 0) {
        elements.newsList.innerHTML = '<div class="news-loading">Chưa có tin tức mới.</div>';
        return;
    }

    const html = newsData.map(item => {
        const sentimentClass = item.sentiment_label === 'Positive' ? 'positive' :
            (item.sentiment_label === 'Negative' ? 'negative' : 'neutral');
        const sentimentBadge = item.sentiment_label === 'Positive' ? '📈 Tích cực' :
            (item.sentiment_label === 'Negative' ? '📉 Tiêu cực' : '⚖️ Trung lập');

        return `
            <div class="news-item">
                <div class="news-meta">
                    <span class="news-source">${item.source}</span>
                    <span class="news-time">${item.date ? new Date(item.date).toLocaleDateString('vi-VN') : ''}</span>
                    <span class="sentiment-badge ${sentimentClass}">${sentimentBadge} (${item.sentiment_score})</span>
                </div>
                <h4 class="news-title"><a href="${item.link}" target="_blank">${item.title}</a></h4>
                <p class="news-summary">${item.summary ? item.summary.substring(0, 100) + '...' : ''}</p>
            </div>
        `;
    }).join('');

    elements.newsList.innerHTML = html;
}

// ===== CALCULATOR =====
function showCalculator() {
    const modal = document.getElementById('calculatorModal');
    if (modal) {
        modal.style.display = 'block';
        // Set default asset
        if (elements.calcAsset) elements.calcAsset.value = state.asset;
        updateCalc();
    }
}

// ===== NEWS SENTIMENT =====
async function fetchNews() {
    try {
        const response = await fetch(`${API_BASE}/api/news?asset=${state.asset}`);
        const data = await response.json();

        if (data.success) {
            updateNewsDisplay(data.news);
        } else {
            console.error('Failed to fetch news:', data.message);
        }
    } catch (error) {
        console.error('Error fetching news:', error);
    }
}

function updateNewsDisplay(newsData) {
    if (!elements.newsList) return;

    if (!newsData || newsData.length === 0) {
        elements.newsList.innerHTML = '<div class="news-loading">Chưa có tin tức mới.</div>';
        return;
    }

    const html = newsData.map(item => {
        const sentimentClass = item.sentiment_label === 'Positive' ? 'positive' :
            (item.sentiment_label === 'Negative' ? 'negative' : 'neutral');
        const sentimentBadge = item.sentiment_label === 'Positive' ? '📈 Tích cực' :
            (item.sentiment_label === 'Negative' ? '📉 Tiêu cực' : '⚖️ Trung lập');

        return `
            <div class="news-item">
                <div class="news-meta">
                    <span class="news-source">${item.source}</span>
                    <span class="news-time">${item.date ? new Date(item.date).toLocaleDateString('vi-VN') : ''}</span>
                    <span class="sentiment-badge ${sentimentClass}">${sentimentBadge} (${item.sentiment_score})</span>
                </div>
                <h4 class="news-title"><a href="${item.link}" target="_blank">${item.title}</a></h4>
                <p class="news-summary">${item.summary ? item.summary.substring(0, 100) + '...' : ''}</p>
            </div>
        `;
    }).join('');

    elements.newsList.innerHTML = html;
}

function updateCalcAsset() {
    updateCalc();
}

// Calc logic
function updateCalc() {
    if (!elements.calcBuyPrice) return;

    const asset = elements.calcAsset.value;
    const buyPrice = parseFloat(elements.calcBuyPrice.value) || 0;
    const amount = parseFloat(elements.calcAmount.value) || 1;

    // Get current price from state
    let currentPriceVND = 0;
    let predPriceVND = 0;

    // We assume state.predictions matches current asset. 
    // If not (e.g. user toggled asset in modal but didn't reload main), we rely on loaded state.
    // Ideally we should warn or fetch. For now, assume state tracks main UI asset.
    if (state.predictions && state.predictions.last_known) {
        if (asset === state.asset) {
            currentPriceVND = state.predictions.last_known.price;
            predPriceVND = state.predictions.predictions[0].price; // Tomorrow
        } else {
            // Fallback or 0
        }
    }

    elements.calcCurrentPrice.textContent = currentPriceVND ? formatPrice(currentPriceVND) + ' VND' : '--';
    elements.calcPredPrice.textContent = predPriceVND ? formatPrice(predPriceVND) + ' VND' : '--';

    if (buyPrice > 0 && currentPriceVND > 0) {
        // Enhanced fee calculation
        const spread = 0.02; // 2% market spread
        const transactionFee = 0.001; // 0.1% per transaction
        const taxRate = 0.10; // 10% capital gains tax

        // Sell Now calculations
        const sellPriceNow = currentPriceVND * (1 - spread / 2);
        const feeNow = sellPriceNow * transactionFee;
        const netSellNow = sellPriceNow - feeNow;
        const profitNow = (netSellNow - buyPrice) * amount;
        const profitNowAfterTax = profitNow > 0 ? profitNow * (1 - taxRate) : profitNow;
        const profitNowPercent = ((netSellNow - buyPrice) / buyPrice) * 100;

        // Update current profit display
        elements.calcProfitNow.textContent = formatPrice(profitNowAfterTax) + ' VND';
        elements.calcProfitNow.className = profitNowAfterTax >= 0 ? 'calc-value positive' : 'calc-value negative';

        // Add percentage display
        const profitNowPercentEl = document.getElementById('calcProfitNowPercent');
        if (profitNowPercentEl) {
            profitNowPercentEl.textContent = `${profitNowPercent >= 0 ? '+' : ''}${profitNowPercent.toFixed(2)}%`;
            profitNowPercentEl.className = profitNowPercent >= 0 ? 'calc-value positive' : 'calc-value negative';
        }

        // Sell Predicted calculations (Day 7)
        const sellPricePred = predPriceVND * (1 - spread / 2);
        const feePred = sellPricePred * transactionFee;
        const netSellPred = sellPricePred - feePred;
        const profitPred = (netSellPred - buyPrice) * amount;
        const profitPredAfterTax = profitPred > 0 ? profitPred * (1 - taxRate) : profitPred;
        const profitPredPercent = ((netSellPred - buyPrice) / buyPrice) * 100;

        // Update predicted profit display
        elements.calcProfitPred.textContent = formatPrice(profitPredAfterTax) + ' VND';
        elements.calcProfitPred.className = profitPredAfterTax >= 0 ? 'calc-value positive' : 'calc-value negative';

        // Add percentage display
        const profitPredPercentEl = document.getElementById('calcProfitPredPercent');
        if (profitPredPercentEl) {
            profitPredPercentEl.textContent = `${profitPredPercent >= 0 ? '+' : ''}${profitPredPercent.toFixed(2)}%`;
            profitPredPercentEl.className = profitPredPercent >= 0 ? 'calc-value positive' : 'calc-value negative';
        }

        // Scenario calculations
        const bestCasePrice = predPriceVND * 1.10; // +10%
        const worstCasePrice = predPriceVND * 0.90; // -10%

        const bestCaseProfit = ((bestCasePrice * (1 - spread / 2) - buyPrice) * amount) * (profitPred > 0 ? (1 - taxRate) : 1);
        const worstCaseProfit = ((worstCasePrice * (1 - spread / 2) - buyPrice) * amount) * (profitPred > 0 ? (1 - taxRate) : 1);

        // Update scenario displays
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

// Add event listeners for calc inputs
if (elements.calcBuyPrice) {
    elements.calcBuyPrice.addEventListener('input', updateCalc);
    elements.calcAmount.addEventListener('input', updateCalc);
}
// GLOBAL MODAL HELPERS
window.showCalculator = showCalculator;



// ========== LOCAL PRICES FETCHING ==========
async function fetchLocalPrices() {
    try {
        const tableBody = document.getElementById('localPriceTableBody');
        const timeBadge = document.getElementById('localPriceTime');
        if (!tableBody) return;

        // Use template literal correctly
        const response = await fetch(`${API_BASE}/api/prices/local`);
        const result = await response.json();

        if (result.success && result.data.items.length > 0) {
            // Expose for Portfolio logic
            window.latestLocalPrices = result.data;
            if (window.updatePortfolioUI) window.updatePortfolioUI();

            // Mock Data Indicator
            const headerBadges = document.querySelector('.local-prices-section .header-badges');
            if (result.data.is_mock) {
                if (!document.getElementById('mockBadge')) {
                    const badge = document.createElement('span');
                    badge.id = 'mockBadge';
                    badge.className = 'badge warning';
                    badge.textContent = '⚠️ Dữ liệu mẫu (Demo)';
                    badge.title = 'Không thể kết nối trực tiếp đến SJC/PNJ/DOJI. Hiển thị dữ liệu giả lập.';
                    badge.style.background = 'rgba(255, 165, 2, 0.2)';
                    badge.style.color = '#ffa502';
                    badge.style.marginLeft = '10px';
                    headerBadges.appendChild(badge);
                }
            } else {
                const existing = document.getElementById('mockBadge');
                if (existing) existing.remove();
            }

            tableBody.innerHTML = '';

            // Check raw items
            console.log("Raw Local Items:", result.data.items);

            // Filter by Asset State
            const filteredItems = result.data.items.filter(item => {
                const brandName = item.brand.toUpperCase();
                const prodName = item.product_type.toUpperCase();

                // Debug log for one item to check content
                // console.log(`Filtering: ${prodName} (${state.asset})`);

                if (state.asset === 'gold') {
                    // Gold Mode
                    // Exclude Silver products
                    if (prodName.includes('BẠC') && !prodName.includes('BẠC LIÊU')) return false;
                    return true;
                } else {
                    // Silver Mode
                    // STRICTLY exclude "VÀNG" to avoid "Vàng SJC - Bạc Liêu"
                    if (prodName.includes('VÀNG')) return false;

                    // Must contain BẠC or SILVER
                    return prodName.includes('BẠC') || prodName.includes('SILVER');
                }
            });

            console.log("Filtered Items:", filteredItems);

            // Sort by lowest spread (best value first)
            const sortedItems = [...filteredItems].sort((a, b) => {
                const spreadA = a.sell_price - a.buy_price;
                const spreadB = b.sell_price - b.buy_price;
                return spreadA - spreadB;
            });

            // Find min/max spread for color scaling
            const spreads = sortedItems.map(i => i.sell_price - i.buy_price);
            const minSpread = Math.min(...spreads);
            const maxSpread = Math.max(...spreads);

            sortedItems.forEach((item, index) => {
                const spread = item.sell_price - item.buy_price;
                const spreadPercent = item.sell_price > 0 ? ((spread / item.sell_price) * 100).toFixed(2) : 0;

                // Color coding: green (low) to red (high)
                let spreadClass = 'spread-medium';
                if (spread === minSpread) {
                    spreadClass = 'spread-low';
                } else if (spread === maxSpread || spreadPercent > 3) {
                    spreadClass = 'spread-high';
                }

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

            // Update Current Price Card with Best Local Price
            if (filteredItems.length > 0) {
                // Priority: SJC for Gold, Bạc miếng Phú Quý for Silver
                let bestItem = filteredItems[0];

                if (state.asset === 'gold') {
                    // Gold: Prefer SJC
                    const sjc = filteredItems.find(i => i.brand.match(/SJC/i));
                    if (sjc) bestItem = sjc;
                } else {
                    // Silver: Prefer "Bạc miếng Phú Quý 999 1 lượng"
                    const phuQuyMieng = filteredItems.find(i =>
                        i.product_type.includes('Bạc miếng') &&
                        i.product_type.includes('Phú Quý') &&
                        i.product_type.includes('1 lượng')
                    );
                    if (phuQuyMieng) {
                        bestItem = phuQuyMieng;
                    } else {
                        // Fallback: any Phú Quý silver product
                        const phuQuy = filteredItems.find(i => i.brand.match(/Phú Quý/i));
                        if (phuQuy) bestItem = phuQuy;
                    }
                }

                console.log(`Best Item for ${state.asset}:`, bestItem);

                // Update the main card
                if (elements.currentPrice) {
                    // Force update price
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

// Auto-init
setTimeout(fetchLocalPrices, 1000);
setInterval(fetchLocalPrices, 300000);

// ========== AI BUY SCORE ==========
async function loadBuyScore() {
    try {
        const asset = state.asset || 'silver';
        const response = await fetch(`${API_BASE}/api/buy-score?asset=${asset}`);
        const data = await response.json();

        if (data.success && data.data) {
            displayBuyScore(data.data);
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

    // Update score directly (don't use animateNumber - it has currency formatting)
    scoreEl.textContent = score;

    // Update label
    if (labelEl) labelEl.textContent = scoreData.label;

    // Update recommendation
    if (recommendationEl) recommendationEl.textContent = scoreData.recommendation;

    // Update asset badge
    if (assetEl) {
        assetEl.textContent = scoreData.asset_type === 'gold' ? 'Vàng' : 'Bạc';
        assetEl.style.background = scoreData.asset_type === 'gold'
            ? 'rgba(255, 215, 0, 0.2)'
            : 'rgba(192, 192, 192, 0.2)';
        assetEl.style.color = scoreData.asset_type === 'gold'
            ? 'var(--gold-primary)'
            : 'var(--silver-primary)';
    }

    // Update card data attribute for color styling
    if (card) {
        if (score >= 60) {
            card.setAttribute('data-score-level', 'high');
        } else if (score >= 40) {
            card.setAttribute('data-score-level', 'medium');
        } else {
            card.setAttribute('data-score-level', 'low');
        }
    }

    // Animate gauge needle (score 0-100 maps to -90deg to +90deg)
    if (ring) {
        const angle = -90 + (score / 100) * 180;
        ring.style.setProperty('--score-angle', angle);
    }

    // Build factors HTML
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
            toggleText.textContent = factorsEl.classList.contains('expanded')
                ? 'Ẩn chi tiết'
                : 'Xem chi tiết';
        }
    }
}

// Load buy score and news after initial load and refresh periodically
setTimeout(() => {
    loadBuyScore();
    fetchNews();
}, 2000);

setInterval(() => {
    loadBuyScore();
    fetchNews();
}, 300000); // Every 5 minutes

// Also reload buy score when asset changes
const originalToggleAsset = typeof toggleAsset === 'function' ? toggleAsset : null;
if (originalToggleAsset) {
    window._originalToggleAssetForBuyScore = originalToggleAsset;
}

// Hook into asset toggle to reload buy score
document.addEventListener('click', function (e) {
    if (e.target.classList.contains('toggle-btn')) {
        setTimeout(() => {
            loadBuyScore();
            fetchNews();
        }, 500);
    }
});

console.log('App.js v2.0.3 Loaded - With AI Buy Score');
