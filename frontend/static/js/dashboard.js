// Dashboard JavaScript - Complete Implementation
class FraudDashboard {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.charts = {};
        this.recentTransactions = [];
        this.stats = {
            totalPredictions: 0,
            fraudDetected: 0,
            fraudRate: '0%',
            mlModels: 0
        };
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.generateTransactionId();
        await this.loadInitialData();
        this.startAutoRefresh();
        this.initCharts();
        this.showWelcomeMessage();
    }

    setupEventListeners() {
        // Transaction form submission
        document.getElementById('transactionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.predictTransaction();
        });

        // Auto-generate transaction ID when amount changes
        document.getElementById('amount').addEventListener('input', () => {
            this.generateTransactionId();
        });

        // Form field validations
        document.getElementById('userId').addEventListener('input', this.validateUserId);
        document.getElementById('amount').addEventListener('input', this.validateAmount);

        // Refresh button
        document.addEventListener('click', (e) => {
            if (e.target.matches('.btn-refresh') || e.target.closest('.btn-refresh')) {
                this.refreshData();
            }
        });

        // Test scenarios buttons
        this.createTestScenarioButtons();
    }

    createTestScenarioButtons() {
        const formContainer = document.querySelector('.transaction-form-container');
        const testScenariosDiv = document.createElement('div');
        testScenariosDiv.className = 'test-scenarios';
        testScenariosDiv.innerHTML = `
            <h4><i class="fas fa-flask"></i> Quick Test Scenarios</h4>
            <div class="scenario-buttons">
                <button type="button" class="btn-scenario low" onclick="dashboard.loadTestScenario('low')">
                    <i class="fas fa-check-circle"></i> Low Risk
                </button>
                <button type="button" class="btn-scenario medium" onclick="dashboard.loadTestScenario('medium')">
                    <i class="fas fa-exclamation-triangle"></i> Medium Risk
                </button>
                <button type="button" class="btn-scenario high" onclick="dashboard.loadTestScenario('high')">
                    <i class="fas fa-ban"></i> High Risk
                </button>
            </div>
        `;
        
        formContainer.appendChild(testScenariosDiv);
    }

    loadTestScenario(riskLevel) {
        const scenarios = {
            low: {
                amount: 45.99,
                paymentMethod: 'debit_card',
                merchantCategory: 'grocery',
                country: 'US',
                userId: 'SAFE_USER_001'
            },
            medium: {
                amount: 750.00,
                paymentMethod: 'credit_card',
                merchantCategory: 'retail',
                country: 'UK',
                userId: 'NORMAL_USER_001'
            },
            high: {
                amount: 5000.00,
                paymentMethod: 'cryptocurrency',
                merchantCategory: 'online',
                country: 'Unknown',
                userId: 'SUSPICIOUS_USER_001'
            }
        };

        const scenario = scenarios[riskLevel];
        if (scenario) {
            document.getElementById('amount').value = scenario.amount;
            document.getElementById('paymentMethod').value = scenario.paymentMethod;
            document.getElementById('merchantCategory').value = scenario.merchantCategory;
            document.getElementById('country').value = scenario.country;
            document.getElementById('userId').value = scenario.userId;
            this.generateTransactionId();
        }
    }

    validateUserId(e) {
        const userId = e.target.value;
        const isValid = userId.length >= 3 && /^[A-Z0-9_]+$/i.test(userId);
        e.target.style.borderColor = isValid ? '#48bb78' : '#f56565';
    }

    validateAmount(e) {
        const amount = parseFloat(e.target.value);
        const isValid = amount > 0 && amount <= 100000;
        e.target.style.borderColor = isValid ? '#48bb78' : '#f56565';
    }

    generateTransactionId() {
        const timestamp = new Date().getTime();
        const random = Math.floor(Math.random() * 1000);
        const transactionId = `TXN_${timestamp}_${random}`;
        document.getElementById('transactionId').value = transactionId;
    }

    async loadInitialData() {
        await this.updateApiStatus();
        await this.updateStats();
        this.loadSampleTransactions();
    }

    async updateApiStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/health`, { timeout: 5000 });
            const statusElement = document.getElementById('apiStatus');
            
            if (response.ok) {
                const data = await response.json();
                statusElement.innerHTML = '<i class="fas fa-circle"></i> Connected';
                statusElement.classList.add('connected');
                
                // Update last updated time
                document.getElementById('lastUpdated').textContent = 
                    `Last Updated: ${new Date().toLocaleTimeString()}`;
                
                return true;
            } else {
                statusElement.innerHTML = '<i class="fas fa-circle"></i> Error';
                statusElement.classList.remove('connected');
                return false;
            }
        } catch (error) {
            const statusElement = document.getElementById('apiStatus');
            statusElement.innerHTML = '<i class="fas fa-circle"></i> Offline';
            statusElement.classList.remove('connected');
            console.error('API Status Error:', error);
            return false;
        }
    }

    async updateStats() {
        try {
            const response = await fetch(`${this.apiUrl}/api/v1/stats`);
            if (response.ok) {
                const stats = await response.json();
                
                // Store stats
                this.stats = {
                    totalPredictions: stats.total_predictions || 0,
                    fraudDetected: stats.fraud_detected || 0,
                    fraudRate: stats.fraud_rate || '0%',
                    mlModels: stats.models_loaded || 0
                };
                
                // Update stat cards with animation
                this.animateStatUpdate('totalPredictions', this.stats.totalPredictions);
                this.animateStatUpdate('fraudDetected', this.stats.fraudDetected);
                document.getElementById('fraudRate').textContent = this.stats.fraudRate;
                document.getElementById('mlModels').textContent = this.stats.mlModels;
                
                // Update charts with new data
                this.updateCharts(stats);
                
                return true;
            }
        } catch (error) {
            console.error('Failed to update stats:', error);
        }
        return false;
    }

    animateStatUpdate(elementId, newValue) {
        const element = document.getElementById(elementId);
        const currentValue = parseInt(element.textContent.replace(/,/g, '')) || 0;
        
        if (currentValue !== newValue) {
            this.animateValue(element, currentValue, newValue, 1000);
        }
    }

    async predictTransaction() {
        const form = document.getElementById('transactionForm');
        const formData = new FormData(form);
        
        // Validate form
        if (!this.validateForm(formData)) {
            this.showNotification('Please fill all required fields correctly', 'error');
            return;
        }
        
        // Show loading state
        const predictBtn = document.querySelector('.btn-predict');
        const originalText = predictBtn.innerHTML;
        predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing Transaction...';
        predictBtn.disabled = true;

        // Hide previous results
        document.getElementById('predictionResult').style.display = 'none';

        // Prepare transaction data
        const transactionData = {
            transaction_id: formData.get('transactionId'),
            user_id: formData.get('userId'),
            merchant_id: `MERCHANT_${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`,
            amount: parseFloat(formData.get('amount')),
            payment_method: formData.get('paymentMethod'),
            merchant_category: formData.get('merchantCategory'),
            country: formData.get('country'),
            device_type: 'desktop',
            timestamp: new Date().toISOString()
        };

        try {
            const startTime = Date.now();
            const response = await fetch(`${this.apiUrl}/api/v1/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(transactionData)
            });

            const endTime = Date.now();
            const processingTime = endTime - startTime;

            if (response.ok) {
                const result = await response.json();
                this.showPredictionResult(result, processingTime);
                this.addRecentTransaction({...result, amount: transactionData.amount});
                this.showNotification('Transaction analyzed successfully!', 'success');
                
                // Refresh stats after prediction
                setTimeout(() => this.updateStats(), 1000);
            } else {
                const errorData = await response.json().catch(() => ({}));
                this.showNotification(`Prediction failed: ${errorData.detail || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            this.showNotification('Connection error. Please check if the API is running.', 'error');
        } finally {
            // Reset button
            predictBtn.innerHTML = originalText;
            predictBtn.disabled = false;
        }
    }

    validateForm(formData) {
        const userId = formData.get('userId');
        const amount = parseFloat(formData.get('amount'));
        
        return userId && userId.length >= 3 && amount > 0 && amount <= 100000;
    }

    showPredictionResult(result, processingTime) {
        const resultDiv = document.getElementById('predictionResult');
        const riskCircle = document.getElementById('riskCircle');
        const riskPercentage = document.getElementById('riskPercentage');
        const riskLevel = document.getElementById('riskLevel');
        const transactionStatus = document.getElementById('transactionStatus');
        const confidence = document.getElementById('confidence');
        const modelUsed = document.getElementById('modelUsed');
        const processingTimeElement = document.getElementById('processingTime');

        // Update risk indicator with animation
        const fraudProb = (result.fraud_probability * 100);
        const displayProb = Math.min(fraudProb, 100).toFixed(1);
        
        // Animate risk percentage
        this.animateRiskIndicator(riskPercentage, riskCircle, fraudProb);
        
        // Update risk level and color
        const riskText = result.risk_level.charAt(0).toUpperCase() + result.risk_level.slice(1) + ' Risk';
        riskLevel.textContent = riskText;
        riskLevel.className = `risk-level ${result.risk_level}`;

        // Update other details
        transactionStatus.textContent = result.status.charAt(0).toUpperCase() + result.status.slice(1);
        transactionStatus.className = `status-badge ${result.status}`;
        
        confidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
        modelUsed.textContent = result.model_used.replace('_', ' ').toUpperCase();
        processingTimeElement.textContent = `${processingTime}ms`;

        // Color code risk circle
        const riskColor = this.getRiskColor(result.risk_level);
        riskCircle.style.background = `conic-gradient(from 0deg, ${riskColor} 0%, ${riskColor} ${fraudProb}%, #e2e8f0 ${fraudProb}%, #e2e8f0 100%)`;

        // Show result with animation
        resultDiv.style.display = 'block';
        resultDiv.style.opacity = '0';
        resultDiv.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            resultDiv.style.transition = 'all 0.3s ease';
            resultDiv.style.opacity = '1';
            resultDiv.style.transform = 'translateY(0)';
        }, 100);

        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    animateRiskIndicator(percentageElement, circleElement, targetValue) {
        let currentValue = 0;
        const duration = 1500;
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            currentValue = targetValue * progress;
            
            percentageElement.textContent = `${currentValue.toFixed(1)}%`;
            circleElement.style.setProperty('--risk-percentage', `${currentValue}%`);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }

    getRiskColor(riskLevel) {
        const colors = {
            low: '#48bb78',
            medium: '#ed8936', 
            high: '#f56565',
            critical: '#c53030'
        };
        return colors[riskLevel] || colors.low;
    }

    addRecentTransaction(result) {
        const transaction = {
            id: result.transaction_id,
            amount: result.amount,
            riskLevel: result.risk_level,
            status: result.status,
            probability: result.fraud_probability,
            time: new Date().toLocaleTimeString()
        };

        this.recentTransactions.unshift(transaction);
        if (this.recentTransactions.length > 15) {
            this.recentTransactions.pop();
        }

        this.updateRecentTransactionsTable();
    }

    updateRecentTransactionsTable() {
        const tbody = document.getElementById('transactionsBody');
        tbody.innerHTML = '';

        this.recentTransactions.forEach((transaction, index) => {
            const row = tbody.insertRow();
            row.style.animationDelay = `${index * 0.1}s`;
            row.className = 'transaction-row';
            
            row.innerHTML = `
                <td>${transaction.id}</td>
                <td>$${transaction.amount.toFixed(2)}</td>
                <td>
                    <span class="risk-badge ${transaction.riskLevel}">
                        ${transaction.riskLevel.toUpperCase()}
                        <small>(${(transaction.probability * 100).toFixed(1)}%)</small>
                    </span>
                </td>
                <td><span class="status-badge ${transaction.status}">${transaction.status.toUpperCase()}</span></td>
                <td>${transaction.time}</td>
            `;
        });
    }

    initCharts() {
        this.initFraudTrendChart();
        this.initRiskDistributionChart();
    }

    initFraudTrendChart() {
        const fraudCtx = document.getElementById('fraudChart').getContext('2d');
        this.charts.fraud = new Chart(fraudCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Fraud Rate %',
                    data: [],
                    borderColor: '#f56565',
                    backgroundColor: 'rgba(245, 101, 101, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Total Predictions',
                    data: [],
                    borderColor: '#4299e1',
                    backgroundColor: 'rgba(66, 153, 225, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Fraud Rate %'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Total Predictions'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }

    initRiskDistributionChart() {
        const riskCtx = document.getElementById('riskChart').getContext('2d');
        this.charts.risk = new Chart(riskCtx, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'],
                datasets: [{
                    data: [70, 20, 8, 2],
                    backgroundColor: ['#48bb78', '#ed8936', '#f56565', '#c53030'],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    updateCharts(stats) {
        this.updateFraudTrendChart(stats);
        this.updateRiskDistributionChart();
    }

    updateFraudTrendChart(stats) {
        if (!this.charts.fraud) return;
        
        const now = new Date().toLocaleTimeString();
        const fraudRate = parseFloat(stats.fraud_rate?.replace('%', '') || 0);
        const totalPredictions = stats.total_predictions || 0;
        
        const chart = this.charts.fraud;
        chart.data.labels.push(now);
        chart.data.datasets[0].data.push(fraudRate);
        chart.data.datasets[1].data.push(totalPredictions);
        
        // Keep only last 15 data points
        if (chart.data.labels.length > 15) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
            chart.data.datasets[1].data.shift();
        }
        
        chart.update('none'); // Update without animation for performance
    }

    updateRiskDistributionChart() {
        if (!this.charts.risk || this.recentTransactions.length === 0) return;
        
        // Calculate risk distribution from recent transactions
        const riskCounts = { low: 0, medium: 0, high: 0, critical: 0 };
        
        this.recentTransactions.forEach(transaction => {
            riskCounts[transaction.riskLevel] = (riskCounts[transaction.riskLevel] || 0) + 1;
        });
        
        const total = this.recentTransactions.length;
        const riskPercentages = [
            Math.round((riskCounts.low / total) * 100),
            Math.round((riskCounts.medium / total) * 100),
            Math.round((riskCounts.high / total) * 100),
            Math.round((riskCounts.critical / total) * 100)
        ];
        
        this.charts.risk.data.datasets[0].data = riskPercentages;
        this.charts.risk.update();
    }

    loadSampleTransactions() {
        // Load some sample transactions for demo
        const samples = [
            { id: 'TXN_SAMPLE_001', amount: 45.99, riskLevel: 'low', status: 'approved', probability: 0.15, time: '10:30:15' },
            { id: 'TXN_SAMPLE_002', amount: 1250.00, riskLevel: 'high', status: 'blocked', probability: 0.85, time: '10:25:30' },
            { id: 'TXN_SAMPLE_003', amount: 299.99, riskLevel: 'medium', status: 'review', probability: 0.35, time: '10:20:45' }
        ];
        
        this.recentTransactions = samples;
        this.updateRecentTransactionsTable();
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
                <span>${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    showWelcomeMessage() {
        setTimeout(() => {
            this.showNotification('Welcome to Fraud Detection Engine Dashboard! Test your transactions above.', 'info');
        }, 2000);
    }

    animateValue(element, start, end, duration) {
        const startTime = Date.now();
        const run = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const current = Math.floor(start + (end - start) * progress);
            element.textContent = current.toLocaleString();
            if (progress < 1) {
                requestAnimationFrame(run);
            }
        };
        requestAnimationFrame(run);
    }

    async refreshData() {
        const refreshBtn = document.querySelector('.btn-refresh i');
        refreshBtn.style.animation = 'spin 1s linear infinite';
        
        await Promise.all([
            this.updateApiStatus(),
            this.updateStats()
        ]);
        
        refreshBtn.style.animation = '';
        this.showNotification('Data refreshed successfully!', 'success');
    }

    startAutoRefresh() {
        // Auto-refresh every 30 seconds
        setInterval(() => {
            this.refreshData();
        }, 30000);
    }
}

// Global dashboard instance
let dashboard;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new FraudDashboard();
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .transaction-row {
        animation: slideInUp 0.3s ease forwards;
        opacity: 0;
        transform: translateY(20px);
    }
    
    @keyframes slideInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        animation: slideInRight 0.3s ease;
    }
    
    .notification.success { border-left: 4px solid #48bb78; }
    .notification.error { border-left: 4px solid #f56565; }
    .notification.info { border-left: 4px solid #4299e1; }
    
    .notification-content {
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .notification-close {
        background: none;
        border: none;
        cursor: pointer;
        opacity: 0.7;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100%);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .test-scenarios {
        margin-top: 20px;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 10px;
    }
    
    .scenario-buttons {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    
    .btn-scenario {
        padding: 8px 16px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    
    .btn-scenario.low { background: #c6f6d5; color: #2f855a; }
    .btn-scenario.medium { background: #feebc8; color: #c05621; }
    .btn-scenario.high { background: #fed7d7; color: #c53030; }
    
    .btn-scenario:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .risk-badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .risk-badge.low { background: #c6f6d5; color: #2f855a; }
    .risk-badge.medium { background: #feebc8; color: #c05621; }
    .risk-badge.high { background: #fed7d7; color: #c53030; }
    .risk-badge.critical { background: #fed7d7; color: #742a2a; }
`;

document.head.appendChild(style);
