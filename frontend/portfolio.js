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

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initial Render
    updatePortfolioUI();

    // Listen for updates
    document.addEventListener('portfolioUpdated', updatePortfolioUI);

    // Form Submit
    const form = document.getElementById('addTxForm');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();

            const asset = form.txAsset.value;
            const brand = document.getElementById('txBrand').value;
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
            alert('Đã thêm giao dịch thành công!');
        });
    }
});

// Expose update function to global scope so app.js can call it when data loads
window.updatePortfolioUI = updatePortfolioUI;

