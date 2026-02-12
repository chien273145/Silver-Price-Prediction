/**
 * Trust Features Module
 * Displays prediction reasoning, market condition assessment, and risk warnings
 * v1.0.0
 */

// ========== PREDICTION REASONING ==========
async function fetchAndDisplayReasoning(apiBase = window.location.origin, currency = 'VND') {
    try {
        const response = await fetch(`${apiBase}/api/predict?currency=${currency}`);
        const data = await response.json();

        if (data.success && data.reasoning) {
            displayReasoning(data.reasoning);
        }
    } catch (error) {
        console.error('Error fetching reasoning:', error);
    }
}

function displayReasoning(reasoning) {
    const container = document.getElementById('predictionReasoning');
    if (!container) return;

    const { direction, confidence, primary_reasons, summary } = reasoning;

    // Direction icon and text
    const directionIcons = {
        'up': '📈',
        'down': '📉',
        'stable': '➡️'
    };

    const directionTexts = {
        'up': 'Tăng',
        'down': 'Giảm',
        'stable': 'Ổn định'
    };

    const confidenceColors = {
        'high': '#2ecc71',
        'medium': '#f1c40f',
        'low': '#e74c3c'
    };

    const confidenceTexts = {
        'high': 'Cao',
        'medium': 'Trung bình',
        'low': 'Thấp'
    };

    let html = `
        <div class="reasoning-header">
            <h3>🔍 Tại Sao Giá Dự Đoán ${directionTexts[direction]}?</h3>
            <div class="confidence-badge" style="background-color: ${confidenceColors[confidence]}">
                Độ tin cậy: ${confidenceTexts[confidence]}
            </div>
        </div>
        <div class="reasoning-summary">
            <p>${summary}</p>
        </div>
        <div class="reasoning-factors">
            <h4>Các yếu tố chính:</h4>
            <div class="factors-grid">
    `;

    primary_reasons.forEach(reason => {
        const impactColors = {
            'very_positive': '#27ae60',
            'positive': '#2ecc71',
            'neutral': '#95a5a6',
            'negative': '#e67e22',
            'very_negative': '#e74c3c'
        };

        html += `
            <div class="factor-card" style="border-left: 4px solid ${impactColors[reason.impact]}">
                <div class="factor-icon">${reason.icon}</div>
                <div class="factor-content">
                    <div class="factor-title">${reason.factor}</div>
                    <div class="factor-detail">${reason.detail}</div>
                </div>
            </div>
        `;
    });

    html += `
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// ========== BUY SCORE & ACTION RECOMMENDATIONS ==========
async function fetchAndDisplayBuyScore(apiBase = window.location.origin, assetType = 'silver', currency = 'VND') {
    try {
        const response = await fetch(`${apiBase}/api/buy-score?asset_type=${assetType}&currency=${currency}`);
        const data = await response.json();

        if (data.success) {
            displayBuyScore(data);
        }
    } catch (error) {
        console.error('Error fetching buy score:', error);
    }
}

function displayBuyScore(data) {
    const container = document.getElementById('buyScoreContainer');
    if (!container) return;

    const {
        buy_score,
        buy_score_label,
        buy_score_color,
        factors,
        market_condition,
        condition_label,
        condition_color,
        educational_points,
        risk_warnings,
        considerations,
        strategies,
        disclaimer
    } = data;

    let html = `
        <!-- Disclaimer Banner -->
        <div class="disclaimer-banner">
            <span class="disclaimer-icon">⚠️</span>
            <p>${disclaimer}</p>
        </div>

        <!-- Buy Score Card -->
        <div class="buy-score-card">
            <div class="score-header">
                <h3>AI Buy Score</h3>
                <div class="score-circle" style="background: linear-gradient(135deg, ${buy_score_color}, ${adjustColor(buy_score_color, -20)})">
                    <span class="score-value">${buy_score}</span>
                    <span class="score-max">/100</span>
                </div>
            </div>
            <div class="score-label" style="color: ${buy_score_color}">${buy_score_label}</div>

            <!-- Factors Breakdown -->
            <div class="factors-breakdown">
                <h4>📊 Phân tích các yếu tố:</h4>
                <div class="factors-list">
    `;

    factors.forEach(factor => {
        const percentage = (factor.points / factor.max) * 100;
        html += `
            <div class="factor-item">
                <div class="factor-header">
                    <span class="factor-icon">${factor.icon}</span>
                    <span class="factor-name">${factor.name}</span>
                    <span class="factor-points">${factor.points}/${factor.max}</span>
                </div>
                <div class="factor-bar">
                    <div class="factor-fill" style="width: ${percentage}%; background-color: ${buy_score_color}"></div>
                </div>
                <div class="factor-detail">${factor.detail}</div>
            </div>
        `;
    });

    html += `
                </div>
            </div>
        </div>

        <!-- Market Condition Assessment -->
        <div class="market-condition-card">
            <h3>📈 Đánh Giá Điều Kiện Thị Trường</h3>
            <div class="condition-badge" style="background-color: ${getConditionColor(condition_color)}">
                ${condition_label}
            </div>

            <!-- Educational Points -->
            <div class="educational-section">
                <h4>💡 Thông Tin Thị Trường:</h4>
                <div class="educational-grid">
    `;

    educational_points.forEach(point => {
        html += `
            <div class="educational-card">
                <div class="edu-icon">${point.icon}</div>
                <div class="edu-content">
                    <div class="edu-title">${point.title}</div>
                    <div class="edu-text">${point.content}</div>
                </div>
            </div>
        `;
    });

    html += `
                </div>
            </div>

            <!-- Risk Warnings -->
            <div class="risk-warnings-section">
                <h4>⚠️ Rủi Ro Cần Lưu Ý:</h4>
                <div class="warnings-list">
    `;

    risk_warnings.forEach(warning => {
        html += `
            <div class="warning-card">
                <span class="warning-icon">${warning.icon}</span>
                <div class="warning-content">
                    <strong>${warning.title}:</strong> ${warning.content}
                </div>
            </div>
        `;
    });

    html += `
                </div>
            </div>

            <!-- Considerations -->
            <div class="considerations-section">
                <h4>🤔 Các Điểm Cần Cân Nhắc:</h4>
                <ul class="considerations-list">
    `;

    considerations.forEach(consideration => {
        html += `
            <li>
                <span class="consideration-icon">${consideration.icon}</span>
                <strong>${consideration.title}:</strong> ${consideration.content}
            </li>
        `;
    });

    html += `
                </ul>
            </div>

            <!-- Investment Strategies (Educational) -->
            <div class="strategies-section">
                <h4>📚 Các Chiến Lược Phổ Biến (Tham Khảo):</h4>
                <div class="strategies-grid">
    `;

    strategies.forEach(strategy => {
        html += `
            <div class="strategy-card">
                <div class="strategy-name">${strategy.name}</div>
                <div class="strategy-desc">${strategy.description}</div>
                <div class="strategy-suitable">
                    <em>Phù hợp với: ${strategy.suitable_for}</em>
                </div>
            </div>
        `;
    });

    html += `
                </div>
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// ========== UTILITY FUNCTIONS ==========
function adjustColor(color, amount) {
    // Simple color adjustment (darken/lighten)
    return color; // Simplified for now
}

function getConditionColor(colorName) {
    const colors = {
        'green': '#2ecc71',
        'lightgreen': '#27ae60',
        'yellow': '#f1c40f',
        'orange': '#e67e22',
        'red': '#e74c3c'
    };
    return colors[colorName] || '#95a5a6';
}

// ========== INITIALIZATION ==========
function initTrustFeatures(assetType = 'silver') {
    const apiBase = window.location.origin;
    const currency = window.state?.currency || 'VND';

    // Fetch and display reasoning
    fetchAndDisplayReasoning(apiBase, currency);

    // Fetch and display buy score
    fetchAndDisplayBuyScore(apiBase, assetType, currency);
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Determine asset type from page
        const isSilverPage = window.location.pathname.includes('silver') || window.location.pathname === '/' || window.location.pathname.includes('index');
        const assetType = isSilverPage ? 'silver' : 'gold';
        initTrustFeatures(assetType);
    });
} else {
    // DOM already loaded
    const isSilverPage = window.location.pathname.includes('silver') || window.location.pathname === '/' || window.location.pathname.includes('index');
    const assetType = isSilverPage ? 'silver' : 'gold';
    initTrustFeatures(assetType);
}

// Export for manual initialization if needed
window.initTrustFeatures = initTrustFeatures;
window.fetchAndDisplayReasoning = fetchAndDisplayReasoning;
window.fetchAndDisplayBuyScore = fetchAndDisplayBuyScore;
