/**
 * Gold Price Prediction Dashboard
 * Gold-Specific JavaScript Application
 * v2.2.0 - Separated from unified app.js
 */

// API Base URL
const API_BASE = window.location.origin;

// Fixed Asset Type - NO TOGGLE
const ASSET = 'gold';

// State
const state = {
    asset: ASSET,  // Fixed to gold
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
        particle.className = 'particle gold';  // Always gold particles

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
// ========== PERFORMANCE TRANSPARENCY ==========
function updatePerformanceDisplay() {
    // Use data embedded in predictions response
    if (!state.predictions || !state.predictions.accuracy_check) return;

    const accuracy = state.predictions.accuracy_check;
    const accuracyContent = document.getElementById('accuracyContent');
    const accuracyBadge = document.getElementById('accuracyBadge');

    if (!accuracyContent) return;

    // accuracy_check returns { date, actual, predicted, diff, diff_pct, accuracy, unit }
    // We map it to the display format

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
                <div class="perf-value">${formatPrice(accuracy.predicted)} <small>tr</small></div>
            </div>
            <div class="perf-item">
                <div class="perf-label">📊 Thực tế (Hôm nay)</div>
                <div class="perf-value">${formatPrice(accuracy.actual)} <small>tr</small></div>
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
            <strong>AI đã dự đoán chính xác ${accuracy.accuracy}% giá vàng hôm nay.</strong>
        </div>
    `;

    accuracyContent.innerHTML = performanceHTML;

    if (accuracyBadge) {
        accuracyBadge.textContent = `${accuracy.accuracy}%`;
        accuracyBadge.style.backgroundColor = accuracy.accuracy >= 95 ? '#00d97e' : (accuracy.accuracy >= 90 ? '#f1c40f' : '#e74c3c');
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
                <div class="perf-label">📅 Date</div>
                <div class="perf-value">${date}</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">🎯 Forecast</div>
                <div class="perf-value">${state.currency === 'VND' ?
            new Intl.NumberFormat('vi-VN').format(forecast.vnd) + ' VND' :
            '$' + forecast.usd}</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">📊 Actual</div>
                <div class="perf-value">${state.currency === 'VND' ?
            new Intl.NumberFormat('vi-VN').format(actual.vnd) + ' VND' :
            '$' + actual.usd}</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">📈 Difference</div>
                <div class="perf-value ${difference.percentage > 0 ? 'positive' : 'negative'}">
                    ${difference.percentage > 0 ? '+' : ''}${difference.percentage}%
                </div>
            </div>
            <div class="perf-item highlight">
                <div class="perf-label">✅ Accuracy</div>
                <div class="perf-value" style="color: ${accuracy.grade_color}">
                    ${accuracy.overall}% (${accuracy.grade})
                </div>
            </div>
            <div class="perf-item">
                <div class="perf-label">💪 Confidence</div>
                <div class="perf-value">${model_confidence}</div>
            </div>
        </div>
        <div class="performance-comment">
            <strong>${accuracy.comment}</strong>
            ${accuracy.direction_correct !== null ?
            `| Direction: ${accuracy.direction_correct ? '✅ Correct' : '❌ Wrong'}` : ''}
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
        // Gold uses gold-specific predict endpoint
        const response = await fetch(`${API_BASE}/api/gold/predict`);
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
    } else {
        factorsHTML = marketDrivers.factors.map(factor =>
            `<div class="factor-item dynamic">
                <h4>📊 Phân tích thị trường</h4>
                <p>${factor}</p>
            </div>`
        ).join('');
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
    metricRMSE: document.getElementById('metricRMSE'),
    metricMAE: document.getElementById('metricMAE'),
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

// API Functions - GOLD SPECIFIC ENDPOINTS
async function fetchAPI(endpoint) {
    try {
        const url = `${API_BASE}${endpoint}${endpoint.includes('?') ? '&' : '?'}t=${Date.now()}`;
        console.log(`[Gold] Fetching: ${url}`);
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
    // Use Vietnam Gold endpoint - Transfer Learning
    const endpoint = `/api/gold-vn/predict`;
    console.log(`[Gold VN] getPredictions endpoint=${endpoint}`);
    return await fetchAPI(endpoint);
}

async function getHistorical() {
    // Use Vietnam Gold endpoint
    const endpoint = `/api/gold-vn/historical?days=${state.historicalDays}`;
    return await fetchAPI(endpoint);
}

async function getRealtime() {
    return await fetchAPI('/api/realtime');
}

async function getModelInfo() {
    // Always use gold endpoint
    return await fetchAPI('/api/gold/metrics');
}

async function getDataStatus() {
    return await fetchAPI('/api/gold/data-status');
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

// Update Functions - Adapted for Vietnam Gold API
function updatePriceCards() {
    if (!state.predictions || !state.predictions.predictions) {
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

    const { predictions, unit } = state.predictions;

    // First prediction = current reference price
    const firstPred = predictions[0];
    const lastPred = predictions[predictions.length - 1];

    // Calculate stats from predictions
    const prices = predictions.map(p => p.predicted_price);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;

    // Current price (first prediction as reference)
    animateNumber(elements.currentPrice, firstPred.predicted_price);
    elements.priceUnit.textContent = unit || 'triệu VND/lượng';

    // Price change from first to last
    const totalChange = lastPred.predicted_price - firstPred.predicted_price;
    const totalChangePct = lastPred.change_percent;

    if (elements.priceChange) {
        const changeValue = elements.priceChange.querySelector('.change-value');
        const changePercent = elements.priceChange.querySelector('.change-percent');
        if (changeValue) {
            changeValue.textContent = formatChange(firstPred.change_percent, true);
            changeValue.className = `change-value ${firstPred.change_percent >= 0 ? 'positive' : 'negative'}`;
        }
        if (changePercent) {
            changePercent.textContent = 'so với hôm nay';
            changePercent.className = 'change-percent';
        }
    }

    // Predicted price (day 7)
    // Predicted price (day 7)
    elements.predictedPrice.textContent = lastPred.predicted_price.toLocaleString('vi-VN');

    // Update range (New)
    const rangeEl = document.getElementById('predictedRange');
    if (rangeEl && lastPred.lower && lastPred.upper) {
        const low = formatPrice(lastPred.lower);
        const high = formatPrice(lastPred.upper);
        rangeEl.textContent = `Vùng an toàn: ${low} - ${high}`;
    } else if (rangeEl) {
        rangeEl.textContent = '';
    }

    // Prediction change
    if (elements.predictionChange) {
        const predChangeValue = elements.predictionChange.querySelector('.change-value');
        const predChangePercent = elements.predictionChange.querySelector('.change-percent');
        if (predChangeValue) {
            predChangeValue.textContent = formatChange(totalChange);
            predChangeValue.className = `change-value ${totalChange >= 0 ? 'positive' : 'negative'}`;
        }
        if (predChangePercent) {
            predChangePercent.textContent = formatChange(totalChangePct, true);
            predChangePercent.className = `change-percent ${totalChangePct >= 0 ? 'positive' : 'negative'}`;
        }
    }

    // Trend badge
    if (elements.trendBadge) {
        const trend = totalChange >= 0 ? 'up' : 'down';
        elements.trendBadge.textContent = trend === 'up' ? '📈 Tăng' : '📉 Giảm';
        elements.trendBadge.className = `trend-badge ${trend}`;
    }

    // Stats
    elements.minPrice.textContent = minPrice.toLocaleString('vi-VN');
    elements.maxPrice.textContent = maxPrice.toLocaleString('vi-VN');
    elements.avgPrice.textContent = avgPrice.toFixed(2).toLocaleString('vi-VN');

    // No exchange rate for VN Gold (already in VND)
    elements.exchangeRate.textContent = 'Giá SJC Việt Nam';

    // Update performance widget (transparency)
    updatePerformanceDisplay();
}

// ===== NEWS SENTIMENT =====
async function fetchNews() {
    try {
        const response = await fetch(`${API_BASE}/api/news?asset=gold`);
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

function updatePredictionTable() {
    if (!state.predictions || !state.predictions.predictions) {
        elements.predictionTable.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 2rem;">Đang tải dữ liệu...</td></tr>';
        return;
    }

    const { predictions } = state.predictions;

    let html = '';
    predictions.forEach((pred) => {
        const trendClass = pred.change_percent >= 0 ? 'trend-up' : 'trend-down';
        const trendIcon = pred.change_percent >= 0 ? '▲' : '▼';

        html += `
            <tr>
                <td>
                    <strong>Ngày ${pred.day}</strong><br>
                    <small style="color: var(--text-muted)">${formatDate(pred.date)}</small>
                </td>
                <td style="font-weight: 600">${pred.predicted_price.toLocaleString('vi-VN')}</td>
                <td class="${trendClass}">${formatChange(pred.change_percent, true)}</td>
                <td class="${trendClass}">${trendIcon}</td>
                <td>triệu VND</td>
            </tr>
        `;
    });

    elements.predictionTable.innerHTML = html;
    let updateText = `Cập nhật: ${new Date().toLocaleTimeString('vi-VN')}`;
    if (state.predictions.model_info && state.predictions.model_info.is_live_prediction) {
        updateText += ` <span style="background: #28a745; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; vertical-align: middle;">LIVE</span>`;
    }
    elements.lastUpdate.innerHTML = updateText;
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

    // VN Gold historical data uses mid_price
    // Filter data based on state.historicalDays
    const allData = state.historical.data;
    const filteredData = allData.slice(-state.historicalDays);

    const historicalData = filteredData.map(item => ({
        x: new Date(item.date),
        y: item.mid_price || item.price
    }));

    // VN Gold predictions use predicted_price
    const lastHistorical = state.historical.data[state.historical.data.length - 1];
    const lastDate = new Date(lastHistorical.date);
    const lastPrice = lastHistorical.mid_price || lastHistorical.price;

    const predictionData = [
        { x: lastDate, y: lastPrice },
        ...state.predictions.predictions.map(item => ({
            x: new Date(item.date),
            y: item.predicted_price
        }))
    ];

    // Prepare confidence data
    const confidenceLower = [
        { x: lastDate, y: lastPrice },
        ...state.predictions.predictions.map(item => ({
            x: new Date(item.date),
            y: item.lower !== undefined ? item.lower : item.predicted_price
        }))
    ];

    const confidenceUpper = [
        { x: lastDate, y: lastPrice },
        ...state.predictions.predictions.map(item => ({
            x: new Date(item.date),
            y: item.upper !== undefined ? item.upper : item.predicted_price
        }))
    ];

    if (state.chart) {
        state.chart.destroy();
    }

    // Gold-specific chart colors
    state.chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Giá lịch sử Vàng',
                    data: historicalData,
                    borderColor: 'rgba(255, 215, 0, 0.8)',
                    backgroundColor: 'rgba(255, 215, 0, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#ffd700',
                    borderWidth: 2
                },
                {
                    label: 'Dự đoán',
                    data: predictionData,
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    borderWidth: 2,
                    order: 1
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
                    backgroundColor: 'rgba(255, 215, 0, 0.15)', // Gold cloud
                    pointRadius: 0,
                    fill: '-1', // Fill to previous dataset (Lower)
                    order: 3
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

// Local Prices
async function fetchLocalPrices() {
    try {
        const response = await fetch(`${API_BASE}/api/prices/local`);
        const data = await response.json();

        if (data.success) {
            window.latestLocalPrices = data.data; // Store for portfolio
            updateLocalPricesTable(data.data);

            // Dispatch event for portfolio update
            document.dispatchEvent(new CustomEvent('portfolioUpdated'));
        }
    } catch (error) {
        console.error('Error fetching local prices:', error);
    }
}

function updateLocalPricesTable(data) {
    const tbody = document.getElementById('localPriceTableBody');
    const timeDisplay = document.getElementById('localPriceTime');

    if (!tbody) return;

    if (!data.items || data.items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="error-cell">Không có dữ liệu giá</td></tr>';
        return;
    }

    // Filter for Gold products only
    const goldItems = data.items.filter(item => {
        const type = item.product_type.toUpperCase();
        return !type.includes('BẠC') && !type.includes('SILVER');
    });

    if (goldItems.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="info-cell">Chưa có dữ liệu giá vàng hôm nay</td></tr>';
        return;
    }

    let html = '';
    goldItems.forEach(item => {
        const buy = item.buy_price;
        const sell = item.sell_price;
        const spread = sell - buy;

        // Brand Badge Color
        let brandBadge = '';
        if (item.brand === 'SJC') brandBadge = 'badge-sjc';
        else if (item.brand === 'DOJI') brandBadge = 'badge-doji';
        else if (item.brand === 'PNJ') brandBadge = 'badge-pnj';

        html += `
            <tr>
                <td><span class="brand-badge ${brandBadge}">${item.brand}</span></td>
                <td>${item.product_type}</td>
                <td class="price-buy">${formatPrice(buy)}</td>
                <td class="price-sell">${formatPrice(sell)}</td>
                <td class="price-spread">${formatPrice(spread)}</td>
            </tr>
        `;
    });

    tbody.innerHTML = html;

    if (timeDisplay && data.updated_at) {
        const date = new Date(data.updated_at);
        timeDisplay.textContent = `Cập nhật: ${date.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}`;
    }
}

// Market Analysis
async function fetchMarketAnalysis() {
    try {
        const response = await fetch(`${API_BASE}/api/market-analysis?asset=gold`);
        const data = await response.json();

        if (data.success && data.analysis) {
            updateMarketAnalysis(data.analysis);
        } else {
            const container = document.getElementById('marketAnalysisContent');
            if (container) container.innerHTML = '<div class="analysis-error">Chưa có dữ liệu phân tích</div>';
        }
    } catch (error) {
        console.error('Error loading market analysis:', error);
        const container = document.getElementById('marketAnalysisContent');
        if (container) container.innerHTML = '<div class="analysis-error">Lỗi kết nối</div>';
    }
}

function updateMarketAnalysis(analysis) {
    const container = document.getElementById('marketAnalysisContent');
    if (!container) return;

    let html = `
        <div class="analysis-summary">
            <div class="analysis-header">
                <span class="analysis-trend ${analysis.trend === 'bullish' ? 'trend-up' : analysis.trend === 'bearish' ? 'trend-down' : 'trend-neutral'}">
                    ${analysis.trend === 'bullish' ? '📈 Xu hướng Tăng' : analysis.trend === 'bearish' ? '📉 Xu hướng Giảm' : '➡️ Đi Ngang'}
                </span>
                <span class="analysis-confidence">Độ tin cậy: ${analysis.confidence}</span>
            </div>
            
            <div class="analysis-rec-box ${analysis.recommendation.includes('MUA') ? 'rec-buy' : analysis.recommendation.includes('BÁN') ? 'rec-sell' : 'rec-hold'}">
                <strong>KHUYẾN NGHỊ:</strong> ${analysis.recommendation}
            </div>
            
            <p class="analysis-text">${analysis.summary}</p>
        </div>
    `;

    // Indicators
    if (analysis.indicators) {
        html += '<div class="analysis-indicators">';
        if (analysis.indicators.vix) {
            const vix = analysis.indicators.vix;
            html += `
                <div class="indicator-item ${vix.impact}">
                    <span class="ind-label">Chỉ số Sợ hãi (VIX):</span>
                    <span class="ind-value">${vix.value} (${vix.status})</span>
                </div>
            `;
        }
        if (analysis.indicators.dxy) {
            const dxy = analysis.indicators.dxy;
            html += `
                <div class="indicator-item ${dxy.impact}">
                    <span class="ind-label">Chỉ số USD (DXY):</span>
                    <span class="ind-value">${dxy.value} (${dxy.status})</span>
                </div>
            `;
        }
        html += '</div>';
    }

    container.innerHTML = html;
}

// Load all data
async function loadData() {
    console.log('🥇 Loading Gold data...');
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
            updatePriceCards();
            updatePredictionTable();
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

        fetchFearGreedIndex();
        fetchPerformanceTransparency();
        fetchMarketFactors();
        loadNews(); // Call loadNews on data refresh
        fetchLocalPrices(); // Fetch local prices
        fetchMarketAnalysis(); // Call fetchMarketAnalysis on data refresh

        showToast('✅ Dữ liệu Vàng đã cập nhật', 'success');

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
    console.log('🥇 Gold Price Prediction Dashboard');
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
        const response = await fetch(`${API_BASE}/api/news?asset=gold`);
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
        // Gold uses gold-specific accuracy endpoint
        const response = await fetch(`${API_BASE}/api/gold/accuracy?t=${Date.now()}`);
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
                    'event_label': 'email_subscription_gold'
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

// ========== LOCAL PRICES FETCHING (GOLD ONLY) ==========
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

            // Filter only GOLD products
            const filteredItems = result.data.items.filter(item => {
                const prodName = item.product_type.toUpperCase();
                // Exclude silver
                if (prodName.includes('BẠC') && !prodName.includes('BẠC LIÊU')) return false;
                return true;
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

            // Update price card with best gold price (SJC)
            if (filteredItems.length > 0) {
                let bestItem = filteredItems[0];
                const sjc = filteredItems.find(i => i.brand.match(/SJC/i));
                if (sjc) bestItem = sjc;

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
        const response = await fetch(`${API_BASE}/api/buy-score?asset=gold`);
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
    scoreEl.textContent = score;

    // Visual Bar Logic
    const visualBar = document.createElement('div');
    visualBar.className = 'score-visual-bar';
    visualBar.innerHTML = `<div class="score-visual-fill" style="width: ${score}%"></div>`;

    // Insert or update visual bar
    const existingBar = scoreEl.parentNode.querySelector('.score-visual-bar');
    if (existingBar) existingBar.replaceWith(visualBar);
    else scoreEl.parentNode.appendChild(visualBar);

    if (labelEl) labelEl.textContent = scoreData.label;
    if (recommendationEl) recommendationEl.textContent = scoreData.recommendation;

    if (assetEl) {
        assetEl.textContent = 'Vàng';
        assetEl.style.background = 'rgba(255, 215, 0, 0.2)';
        assetEl.style.color = 'var(--gold-primary)';
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
                    <span class="factor-points">${factor.points > 0 ? '+' + factor.points : factor.points}</span>
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

setTimeout(() => {
    loadBuyScore();
    fetchNews();
}, 2000);

setInterval(() => {
    loadBuyScore();
    fetchNews();
}, 300000);

console.log('🥇 app-gold.js v2.3.0 - Vietnam SJC Gold');
console.log('🇻🇳 Transfer Learning Model: R²=0.9792, MAPE=1.21%');

