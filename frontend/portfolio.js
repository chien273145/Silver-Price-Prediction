/**
 * Portfolio Manager
 * Handles local storage of assets and P/L calculations
 */

class PortfolioManager {
    constructor() {
        this.STORAGE_KEY = 'silver_price_portfolio_v1';
        this.transactions = this.loadData();
    }

    // Load from LocalStorage
    loadData() {
        try {
            const data = localStorage.getItem(this.STORAGE_KEY);
            return data ? JSON.parse(data) : [];
        } catch (e) {
            console.error('Error loading portfolio:', e);
            return [];
        }
    }

    // Save to LocalStorage
    saveData() {
        try {
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.transactions));
            // Trigger custom event for UI updates
            document.dispatchEvent(new CustomEvent('portfolioUpdated'));
        } catch (e) {
            console.error('Error saving portfolio:', e);
            alert('Không thể lưu dữ liệu (Bộ nhớ đầy?)');
        }
    }

    /**
     * Add a new transaction
     * @param {Object} transaction { asset, brand, price, amount, date, type }
     */
    addTransaction(transaction) {
        const newTx = {
            id: Date.now().toString(), // Simple unique ID
            timestamp: new Date().toISOString(),
            ...transaction
        };
        this.transactions.push(newTx);
        this.saveData();
        return newTx;
    }

    /**
     * Remove a transaction by ID
     */
    removeTransaction(id) {
        this.transactions = this.transactions.filter(tx => tx.id !== id);
        this.saveData();
    }

    /**
     * Calculate Summary based on current market prices
     * @param {Object} currentPrices map of { "SJC": { buy: X, sell: Y }, ... }
     */
    calculatePortfolio(currentPrices) {
        let totalInvested = 0;
        let currentValue = 0;

        this.transactions.forEach(tx => {
            // 1. Calculate Investment
            const cost = tx.price * tx.amount;
            totalInvested += cost;

            // 2. Calculate Current Value
            // Find matching current price (fallback to purchase price if no data)
            let marketPrice = tx.price;

            // Try to find real price
            // currentPrices format expected: { items: [ { brand: 'SJC', buy_price: ..., sell_price: ... } ] }
            if (currentPrices && currentPrices.items) {
                // Filter by asset type first
                const assetItems = currentPrices.items.filter(item => {
                    const prodName = item.product_type.toUpperCase();
                    if (tx.asset === 'silver') {
                        // For silver, product must contain "BẠC" or "SILVER"
                        return (prodName.includes('BẠC') || prodName.includes('SILVER')) && !prodName.includes('VÀNG');
                    } else {
                        // For gold, exclude silver products
                        return !prodName.includes('BẠC') || prodName.includes('BẠC LIÊU');
                    }
                });

                // Now match by brand within the filtered asset type
                const match = assetItems.find(item =>
                    item.brand.toUpperCase() === tx.brand.toUpperCase()
                );

                if (match) {
                    marketPrice = match.buy_price; // Value is what dealer pays you
                } else if (assetItems.length > 0) {
                    // Fallback: first item of correct asset type
                    marketPrice = assetItems[0].buy_price;
                }
            }

            currentValue += marketPrice * tx.amount;
        });

        const profitLoss = currentValue - totalInvested;
        const profitPercent = totalInvested > 0 ? (profitLoss / totalInvested) * 100 : 0;

        return {
            totalInvested,
            currentValue,
            profitLoss,
            profitPercent,
            count: this.transactions.length
        };
    }

    getTransactions() {
        return this.transactions.sort((a, b) => new Date(b.date) - new Date(a.date));
    }
}

// Global instance
window.portfolioManager = new PortfolioManager();

// ========== UI LOGIC ==========

function updatePortfolioUI() {
    const tableBody = document.getElementById('portfolioTableBody');
    const summaryInvested = document.getElementById('pTotalInvested');
    const summaryCurrent = document.getElementById('pCurrentValue');
    const summaryPL = document.getElementById('pProfitLoss');
    const summaryPercent = document.getElementById('pProfitPercent');

    if (!tableBody) return;

    // Get current prices from global state (if available)
    // We expect state.prices to be populated by app.js from /api/prices/local
    // But since app.js handles fetching, we might need a way to access it.
    // For now, let's look at the DOM table or wait for 'portfolioUpdated' event.

    // Better: let's try to get prices from the local price table in DOM if state is not exposed properly yet
    // Or simpler: We'll assume app.js will call this function when data loads.

    // Mock prices if not loaded yet
    const currentPrices = window.latestLocalPrices || { items: [] };

    const summary = portfolioManager.calculatePortfolio(currentPrices);
    const transactions = portfolioManager.getTransactions();

    // Update Summary
    if (summaryInvested) summaryInvested.textContent = formatMoney(summary.totalInvested);
    if (summaryCurrent) summaryCurrent.textContent = formatMoney(summary.currentValue);

    if (summaryPL) {
        summaryPL.textContent = formatMoney(summary.profitLoss);
        summaryPL.className = `p-value ${summary.profitLoss >= 0 ? 'predict-up' : 'predict-down'}`;
    }

    if (summaryPercent) {
        summaryPercent.textContent = `(${summary.profitPercent >= 0 ? '+' : ''}${summary.profitPercent.toFixed(2)}%)`;
        summaryPercent.className = `p-percent ${summary.profitPercent >= 0 ? 'positive' : 'negative'}`;
    }

    // Update Table
    tableBody.innerHTML = '';

    if (transactions.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="8" class="empty-cell">Chưa có giao dịch nào. Hãy thêm giao dịch đầu tiên!</td></tr>';
        return;
    }

    transactions.forEach(tx => {
        // Find current price for this tx - MUST filter by asset type first
        let marketPrice = tx.price;
        let priceSource = 'Gốc';

        if (currentPrices.items) {
            // Filter by asset type first
            const assetItems = currentPrices.items.filter(item => {
                const prodName = item.product_type.toUpperCase();
                if (tx.asset === 'silver') {
                    return (prodName.includes('BẠC') || prodName.includes('SILVER')) && !prodName.includes('VÀNG');
                } else {
                    return !prodName.includes('BẠC') || prodName.includes('BẠC LIÊU');
                }
            });

            const match = assetItems.find(item => item.brand.toUpperCase() === tx.brand.toUpperCase());
            if (match) {
                marketPrice = match.buy_price;
                priceSource = 'Thị trường';
            } else if (assetItems.length > 0) {
                // Fallback: first item of correct asset type
                marketPrice = assetItems[0].buy_price;
                priceSource = assetItems[0].brand;
            }
        }

        const currentValue = marketPrice * tx.amount;
        const pl = currentValue - (tx.price * tx.amount);
        const plPercent = (pl / (tx.price * tx.amount)) * 100;

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${new Date(tx.timestamp).toLocaleDateString('vi-VN')}</td>
            <td><span class="badge" style="background: ${tx.asset === 'gold' ? '#ffd70033' : '#c0c0c033'}; color: ${tx.asset === 'gold' ? '#ffd700' : '#c0c0c0'}">${tx.asset === 'gold' ? 'Vàng' : 'Bạc'}</span></td>
            <td>${tx.brand}</td>
            <td>${tx.amount}</td>
            <td>${formatMoney(tx.price)}</td>
            <td>${formatMoney(marketPrice)}</td>
            <td class="${pl >= 0 ? 'trend-up' : 'trend-down'}">
                ${formatMoney(pl)}<br>
                <small>(${plPercent >= 0 ? '+' : ''}${plPercent.toFixed(1)}%)</small>
            </td>
            <td>
                <button class="delete-btn" onclick="portfolioManager.removeTransaction('${tx.id}')">🗑️</button>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

// Helpers
function formatMoney(amount) {
    return new Intl.NumberFormat('vi-VN', { style: 'currency', currency: 'VND' }).format(amount);
}

function showAddTransaction() {
    const modal = document.getElementById('addTxModal');
    if (modal) modal.style.display = 'block';

    // Set default date to today
    document.getElementById('txDate').valueAsDate = new Date();
}

// Brand Options Logic
const brandSelect = document.getElementById('txBrand');
const assetRadios = document.getElementsByName('txAsset');

const brandOptions = {
    gold: [
        { value: 'SJC', text: 'SJC' },
        { value: 'DOJI', text: 'DOJI' },
        { value: 'WORLD', text: 'Thế Giới (USD)' },
        { value: 'BTMC', text: 'Bảo Tín Minh Châu' },
        { value: 'BTMH', text: 'Bảo Tín Mạnh Hải' },
        { value: 'PHUQUY', text: 'Phú Quý' }
    ],
    silver: [
        { value: 'BTMC', text: 'Bảo Tín Minh Châu' },
        { value: 'PHUQUY', text: 'Phú Quý' },
        { value: 'KNP', text: 'Kim Ngân Phúc' }
    ]
};

function updateBrandOptions() {
    if (!brandSelect) return;

    // Get selected asset
    let selectedAsset = 'gold';
    for (const radio of assetRadios) {
        if (radio.checked) {
            selectedAsset = radio.value;
            break;
        }
    }

    // Clear existing options
    brandSelect.innerHTML = '';

    // Add new options
    const options = brandOptions[selectedAsset] || brandOptions.gold;
    options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt.value;
        option.textContent = opt.text;
        brandSelect.appendChild(option);
    });
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initial Render
    updatePortfolioUI();

    // Initialize brand options
    updateBrandOptions();

    // Listen for updates
    document.addEventListener('portfolioUpdated', updatePortfolioUI);

    // Listen for asset change
    if (assetRadios) {
        for (const radio of assetRadios) {
            radio.addEventListener('change', updateBrandOptions);
        }
    }

    // Form Submit
    const form = document.getElementById('addTxForm');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();

            const asset = form.txAsset.value;
            const brand = document.getElementById('txBrand').value;
            // ... (rest of the submit handler is unchanged, just ensuring context)
            const date = document.getElementById('txDate').value;
            const amount = parseFloat(document.getElementById('txAmount').value);
            const price = parseFloat(document.getElementById('txPrice').value);

            if (amount <= 0 || price <= 0) {
                alert('Vui lòng nhập số lượng và giá hợp lệ');
                return;
            }

            portfolioManager.addTransaction({
                asset, brand, date, amount, price, type: 'buy'
            });

            closeModal('addTxModal');
            form.reset();
            // Reset brands to default (Gold) after reset if needed, or keep current
            // Since reset() might reset radio to default (Gold), we should update options
            setTimeout(updateBrandOptions, 0);
            alert('Đã thêm giao dịch thành công!');
        });
    }
});

// Expose update function to global scope so app.js can call it when data loads
window.updatePortfolioUI = updatePortfolioUI;

// ============================================
// AI Time Machine Functions
// ============================================

/**
 * Load Time Machine predictions from API
 */
async function loadTimeMachine() {
    const container = document.getElementById('timeMachinePredictions');
    if (!container) return;

    container.innerHTML = '<div class="tm-loading">Đang tính toán dự báo tương lai...</div>';

    try {
        // Get portfolio items
        const portfolioItems = portfolioManager.transactions.map(tx => ({
            id: tx.id,
            asset_type: tx.asset,
            brand: tx.brand,
            quantity: parseFloat(tx.amount) || 0,
            buy_price: parseFloat(tx.price) || 0,
            buy_date: tx.date,
        }));

        if (portfolioItems.length === 0) {
            container.innerHTML = '<div class="tm-loading">Thêm tài sản vào portfolio để xem dự báo tương lai</div>';
            return;
        }

        // Get current prices from local prices if available
        let currentGoldPrice = 0;
        let currentSilverPrice = 0;
        if (window.latestLocalPrices && window.latestLocalPrices.items) {
            const goldItem = window.latestLocalPrices.items.find(i => !i.product_type.toUpperCase().includes('BẠC'));
            const silverItem = window.latestLocalPrices.items.find(i => i.product_type.toUpperCase().includes('BẠC'));
            if (goldItem) currentGoldPrice = goldItem.sell_price;
            if (silverItem) currentSilverPrice = silverItem.sell_price;
        }

        const response = await fetch(`${window.location.origin}/api/time-machine`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                portfolio_items: portfolioItems,
                current_gold_price: currentGoldPrice,
                current_silver_price: currentSilverPrice
            })
        });

        const data = await response.json();

        if (data.success) {
            displayTimeMachine(data);
        } else {
            container.innerHTML = '<div class="tm-loading">Không thể tải dự báo</div>';
        }
    } catch (error) {
        console.error('Time Machine error:', error);
        container.innerHTML = '<div class="tm-loading">Lỗi kết nối</div>';
    }
}

/**
 * Display Time Machine predictions
 */
function displayTimeMachine(data) {
    const container = document.getElementById('timeMachinePredictions');
    if (!container) return;

    if (!data.predictions || data.predictions.length === 0) {
        container.innerHTML = '<div class="tm-loading">Không có dữ liệu dự báo</div>';
        return;
    }

    const formatVND = (value) => {
        if (value >= 1_000_000_000) {
            return (value / 1_000_000_000).toFixed(2) + ' tỷ';
        } else if (value >= 1_000_000) {
            return (value / 1_000_000).toFixed(1) + ' tr';
        } else {
            return new Intl.NumberFormat('vi-VN').format(value);
        }
    };

    const html = data.predictions.map(pred => {
        const isPositive = pred.change_percent >= 0;
        const cardClass = isPositive ? 'up' : 'down';
        const changeClass = isPositive ? 'positive' : 'negative';
        const changeSign = isPositive ? '+' : '';

        return `
            <div class="tm-prediction-card ${cardClass}">
                <div class="tm-days">${pred.days} ngày (${pred.date})</div>
                <div class="tm-value">${formatVND(pred.predicted_value)} ₫</div>
                <div class="tm-change ${changeClass}">
                    ${changeSign}${formatVND(pred.change_amount)} (${changeSign}${pred.change_percent}%)
                </div>
                <div class="tm-confidence">
                    ± ${formatVND(pred.confidence_max - pred.predicted_value)}
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = html;
}

// Expose to global scope
window.loadTimeMachine = loadTimeMachine;

// Auto-load Time Machine after portfolio loads
document.addEventListener('portfolioUpdated', () => {
    setTimeout(loadTimeMachine, 500);
    updateSavingsComparison();
    updateAllocationChart();
});

// Initial load after 3 seconds (after prices are fetched)
setTimeout(loadTimeMachine, 3000);


// ============================================
// Savings Comparison Calculator
// ============================================

const BANK_INTEREST_RATE = 0.05; // 5% annual

function updateSavingsComparison() {
    const roiEl = document.getElementById('portfolioROI');
    const bankEl = document.getElementById('bankInterest');
    const resultEl = document.getElementById('savingsResult');

    if (!roiEl || !resultEl) return;

    const transactions = portfolioManager.transactions;
    if (transactions.length === 0) {
        resultEl.textContent = 'Thêm giao dịch để so sánh';
        resultEl.className = 'savings-result';
        return;
    }

    // Calculate portfolio stats
    let totalInvested = 0;
    let totalCurrentValue = 0;
    let totalDays = 0;

    transactions.forEach(tx => {
        const invested = (parseFloat(tx.amount) || 0) * (parseFloat(tx.price) || 0);
        totalInvested += invested;

        // Get current price from window.latestLocalPrices
        const currentPrice = getCurrentPriceForTransaction(tx);
        const currentValue = (parseFloat(tx.amount) || 0) * currentPrice;
        totalCurrentValue += currentValue;

        // Calculate days since purchase
        if (tx.date) {
            const buyDate = new Date(tx.date);
            const today = new Date();
            const days = Math.floor((today - buyDate) / (1000 * 60 * 60 * 24));
            totalDays = Math.max(totalDays, days);
        }
    });

    if (totalInvested === 0) {
        resultEl.textContent = 'Thêm giao dịch để so sánh';
        resultEl.className = 'savings-result';
        return;
    }

    // Calculate actual ROI
    const actualROI = ((totalCurrentValue - totalInvested) / totalInvested) * 100;

    // Calculate what bank savings would give
    const years = Math.max(totalDays / 365, 0.01);
    const bankGains = totalInvested * BANK_INTEREST_RATE * years;
    const bankROI = (bankGains / totalInvested) * 100;

    // Display
    const formatPercent = (val) => (val >= 0 ? '+' : '') + val.toFixed(2) + '%';

    roiEl.textContent = formatPercent(actualROI);
    roiEl.style.color = actualROI >= 0 ? 'var(--success)' : 'var(--danger)';

    bankEl.textContent = formatPercent(bankROI);

    // Compare result
    const diff = actualROI - bankROI;
    if (diff > 0) {
        resultEl.textContent = `🎉 Vượt trội hơn tiết kiệm ${diff.toFixed(2)}%`;
        resultEl.className = 'savings-result positive';
    } else {
        resultEl.textContent = `📉 Thua tiết kiệm ${Math.abs(diff).toFixed(2)}%`;
        resultEl.className = 'savings-result negative';
    }
}

function getCurrentPriceForTransaction(tx) {
    if (!window.latestLocalPrices || !window.latestLocalPrices.items) return 0;

    const assetType = tx.asset;
    const brand = tx.brand;

    const items = window.latestLocalPrices.items.filter(item => {
        const prodName = item.product_type.toUpperCase();
        if (assetType === 'gold') {
            return !prodName.includes('BẠC') || prodName.includes('BẠC LIÊU');
        } else {
            return (prodName.includes('BẠC') || prodName.includes('SILVER')) && !prodName.includes('VÀNG');
        }
    });

    // Try to match brand
    const match = items.find(i => i.brand.toLowerCase() === brand.toLowerCase());
    if (match) return match.sell_price;

    // Fallback to first item
    return items.length > 0 ? items[0].sell_price : 0;
}


// ============================================
// Allocation Pie Chart
// ============================================

let allocationChart = null;

function updateAllocationChart() {
    const canvas = document.getElementById('allocationChart');
    if (!canvas) return;

    const transactions = portfolioManager.transactions;

    if (transactions.length === 0 || typeof Chart === 'undefined') {
        return;
    }

    // Calculate allocation by asset type
    const allocation = { gold: 0, silver: 0 };

    transactions.forEach(tx => {
        const value = (parseFloat(tx.amount) || 0) * (parseFloat(tx.price) || 0);
        const asset = tx.asset || 'silver';
        allocation[asset] = (allocation[asset] || 0) + value;
    });

    const total = allocation.gold + allocation.silver;
    if (total === 0) return;

    const data = {
        labels: ['Vàng', 'Bạc'],
        datasets: [{
            data: [allocation.gold, allocation.silver],
            backgroundColor: ['#FFD700', '#C0C0C0'],
            borderColor: ['rgba(255, 215, 0, 0.8)', 'rgba(192, 192, 192, 0.8)'],
            borderWidth: 2
        }]
    };

    const config = {
        type: 'pie',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#94A3B8',
                        boxWidth: 12,
                        padding: 8
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const value = context.raw;
                            const percent = ((value / total) * 100).toFixed(1);
                            return ` ${new Intl.NumberFormat('vi-VN').format(value)} ₫ (${percent}%)`;
                        }
                    }
                }
            }
        }
    };

    if (allocationChart) {
        allocationChart.destroy();
    }

    allocationChart = new Chart(canvas, config);
}

// Expose functions
window.updateSavingsComparison = updateSavingsComparison;
window.updateAllocationChart = updateAllocationChart;

// Initial load
setTimeout(() => {
    updateSavingsComparison();
    updateAllocationChart();
}, 3500);
