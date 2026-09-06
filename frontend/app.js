// ── State ──────────────────────────────────────────────────────────────────
const STATE = { IDLE: 'idle', CHATTING: 'chatting', INVESTING: 'investing', DONE: 'done' };
let appState = STATE.IDLE;
let isDarkMode = true;
let currentChartsData = {};
let currentRecommendations = [];
let activeStockSymbol = null; // Explicit current stock context (Fix for Item #3)
let chatHistory = []; // Full message history array (Fix for Item #1)
let allocationChartInstance = null;
let priceEmaChartInstance = null;
let rsiChartInstance = null;
let macdChartInstance = null;
let peersChartInstance = null;

// NSE symbol lookup — common names + direct ticker input
const SYMBOL_MAP = {
    'reliance': 'RELIANCE.NS', 'ril': 'RELIANCE.NS',
    'tcs': 'TCS.NS', 'tata consultancy': 'TCS.NS',
    'infosys': 'INFY.NS', 'infy': 'INFY.NS',
    'hdfc bank': 'HDFCBANK.NS', 'hdfcbank': 'HDFCBANK.NS', 'hdfc': 'HDFCBANK.NS',
    'icici': 'ICICIBANK.NS', 'icici bank': 'ICICIBANK.NS', 'icicicbank': 'ICICIBANK.NS',
    'tata motors': 'TATAMOTORS.NS', 'tatamotors': 'TATAMOTORS.NS',
    'airtel': 'BHARTIARTL.NS', 'bharti': 'BHARTIARTL.NS', 'bhartiartl': 'BHARTIARTL.NS',
    'sbi': 'SBIN.NS', 'state bank': 'SBIN.NS', 'sbin': 'SBIN.NS',
    'axis bank': 'AXISBANK.NS', 'axis': 'AXISBANK.NS', 'axisbank': 'AXISBANK.NS',
    'itc': 'ITC.NS',
    'wipro': 'WIPRO.NS',
    'bajaj finance': 'BAJFINANCE.NS', 'bajfinance': 'BAJFINANCE.NS',
    'maruti': 'MARUTI.NS', 'maruti suzuki': 'MARUTI.NS',
    'lt': 'LT.NS', 'larsen': 'LT.NS', 'l&t': 'LT.NS',
    'hcl': 'HCLTECH.NS', 'hcltech': 'HCLTECH.NS',
    'sun pharma': 'SUNPHARMA.NS', 'sunpharma': 'SUNPHARMA.NS',
    'titan': 'TITAN.NS',
    'ultratech': 'ULTRACEMCO.NS', 'ultracemco': 'ULTRACEMCO.NS',
    'asian paints': 'ASIANPAINT.NS', 'asianpaint': 'ASIANPAINT.NS',
    'kotak': 'KOTAKBANK.NS', 'kotakbank': 'KOTAKBANK.NS',
    'paytm': 'PAYTM.NS', 'one97': 'PAYTM.NS',
    'zomato': 'ZOMATO.NS',
    'tata steel': 'TATASTEEL.NS', 'tatasteel': 'TATASTEEL.NS',
    'coal india': 'COALINDIA.NS', 'coalindia': 'COALINDIA.NS',
    'hal': 'HAL.NS', 'bel': 'BEL.NS', 'dlf': 'DLF.NS',
};

const SECTOR_KEYWORDS = {
    banking: ['bank', 'banking', 'finance', 'nbfc', 'financial'],
    it: ['it', 'tech', 'software', 'technology', 'infosys', 'tcs', 'wipro', 'hcl'],
    auto: ['auto', 'automobile', 'ev', 'car', 'vehicle'],
    pharma: ['pharma', 'healthcare', 'medicine', 'drug'],
    fmcg: ['fmcg', 'consumer', 'retail', 'paint'],
    energy: ['energy', 'oil', 'reliance', 'gas'],
};

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initCharts();
    document.getElementById('chat-input').addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleChatSend(); }
    });
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
    // Greeting
    setTimeout(() => {
        appendAIMessage(
            `Good ${getTimeOfDay()}. I'm your NSE intraday trading assistant. You can ask me about any stock, market conditions, or sector trends. When you're ready to invest, just tell me your amount and I'll generate a full allocation plan.`,
            null
        );
    }, 400);
});

function getTimeOfDay() {
    const h = new Date().getHours();
    if (h < 12) return 'morning';
    if (h < 17) return 'afternoon';
    return 'evening';
}

// ── Theme ──────────────────────────────────────────────────────────────────
function initTheme() {
    const saved = localStorage.getItem('nse-theme');
    isDarkMode = saved !== 'light';
    applyTheme();
}
function toggleTheme() {
    isDarkMode = !isDarkMode;
    localStorage.setItem('nse-theme', isDarkMode ? 'dark' : 'light');
    applyTheme();
    updateChartColors();
}
function applyTheme() {
    document.documentElement.classList.toggle('dark', isDarkMode);
    document.getElementById('theme-label').textContent = isDarkMode ? 'Light Mode' : 'Dark Mode';
}

// ── Intent Detection (Extract stock FIRST, protect sector & command keywords) ───
function detectIntent(msg) {
    const lower = msg.toLowerCase();

    // Amount pattern: ₹50000 or 50000 or 50k
    const amountMatch = lower.match(/(?:₹|rs\.?\s*|inr\s*)(\d[\d,]*(?:k)?)|(\d[\d,]*(?:k)?)\s*(?:rupees?|rs|inr)/i)
                     || lower.match(/invest\s+(?:₹|rs\.?\s*)?(\d[\d,]*k?)/i);

    // Invest intent
    const investWords = ['invest', 'allocate', 'put money', 'trade with', 'use', 'buy stocks', 'ready to invest', 'want to invest'];
    const isInvest = investWords.some(w => lower.includes(w)) || !!amountMatch;

    // Chart request
    const chartWords = ['chart', 'graph', 'technical', 'indicator', 'show me', 'plot', 'visuali', 'draw', '5 ema', 'ema', 'rsi', 'macd', 'vwap'];
    const isChart = chartWords.some(w => lower.includes(w));

    // Sector detection — BEFORE DYNAMIC TICKER EXTRACTION
    let detectedSector = null;
    for (const [sector, words] of Object.entries(SECTOR_KEYWORDS)) {
        if (words.some(w => lower.includes(w))) { detectedSector = sector; break; }
    }

    // Stock detection
    let detectedSymbol = null;

    // 1. Direct SYMBOL_MAP lookup
    for (const [name, sym] of Object.entries(SYMBOL_MAP)) {
        if (lower.includes(name)) { detectedSymbol = sym; break; }
    }

    // 2. Direct .NS / .BO tickers
    if (!detectedSymbol) {
        const directTicker = lower.match(/\b([a-z0-9\-]+)\.(ns|bo)\b/i);
        if (directTicker) {
            detectedSymbol = directTicker[0].toUpperCase();
        }
    }

    // 3. Dynamic phrase extraction (only if NOT a sector comparison query)
    if (!detectedSymbol && !detectedSector) {
        const words = lower.replace(/[^a-z0-9\s]/g, ' ').split(/\s+/);
        const ignoreWords = [
            'the', 'a', 'an', 'what', 'how', 'why', 'is', 'my', 'this', 'that', 'for', 'me',
            'explain', 'show', 'tell', 'about', 'chart', 'charts', 'graph', 'indicator',
            'indicators', 'ema', 'rsi', 'macd', 'vwap', 'stock', 'stocks', 'price', 'trend',
            'compare', 'comparison', 'sector', 'sectors', 'which', 'top', 'best', 'market',
            'good', 'strong', 'weak', 'look', 'looking', 'like', 'today'
        ];
        for (const w of words) {
            if (w.length >= 3 && !ignoreWords.includes(w)) {
                if (SYMBOL_MAP[w]) {
                    detectedSymbol = SYMBOL_MAP[w];
                    break;
                } else {
                    detectedSymbol = `${w.toUpperCase()}.NS`;
                    break;
                }
            }
        }
    }

    // Market query
    const marketWords = ['market', 'nse', 'bse', 'sensex', 'nifty', "what's happening", 'today', 'overview', 'condition'];
    const isMarket = marketWords.some(w => lower.includes(w)) && !detectedSymbol && !detectedSector;

    // Extract amount
    let amount = null;
    if (amountMatch) {
        const raw = (amountMatch[1] || amountMatch[2] || '').replace(/,/g, '');
        amount = raw.endsWith('k') ? parseFloat(raw) * 1000 : parseFloat(raw);
    }

    if (isInvest) return { type: 'invest', symbol: detectedSymbol, sector: detectedSector, amount };
    if (isChart)  return { type: 'chart', symbol: detectedSymbol };
    if (detectedSector && lower.includes('compare')) return { type: 'sector', sector: detectedSector };
    if (detectedSymbol) return { type: 'stock', symbol: detectedSymbol };
    if (detectedSector) return { type: 'sector', sector: detectedSector };
    if (isMarket)       return { type: 'market' };
    return { type: 'general', symbol: detectedSymbol };
}



// ── Chat Send ──────────────────────────────────────────────────────────────
async function handleChatSend() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;

    hideSuggestions();
    appendUserMessage(msg);
    input.value = '';

    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = true;

    const typingEl = showTyping();
    const intent = detectIntent(msg);

    try {
        switch (intent.type) {
            case 'stock':   await handleStockIntent(intent, typingEl); break;
            case 'chart':   await handleChartIntent(intent, typingEl); break;
            case 'invest':  await handleInvestIntent(intent, typingEl); break;
            case 'sector':  await handleSectorIntent(intent, typingEl); break;
            case 'market':  await handleMarketIntent(typingEl); break;
            default:        await handleGeneralIntent(msg, typingEl); break;
        }
    } finally {
        sendBtn.disabled = false;
    }
}

function sendSuggestion(text) {
    document.getElementById('chat-input').value = text;
    handleChatSend();
}


function updateSuggestionChips(chips) {
    const s = document.getElementById('suggestions-bar');
    if (!s || !chips || !chips.length) return;
    s.style.display = 'flex';
    s.innerHTML = chips.map(c => `<button class="chip" onclick="sendSuggestion('${c.replace(/'/g, "\\'")}')">${c}</button>`).join('');
}

function hideSuggestions() {
    // Kept for fallback but chips now dynamically update instead of hiding
}


// ── Intent Handlers ────────────────────────────────────────────────────────
async function handleStockIntent(intent, typingEl) {
    const sym = intent.symbol;
    try {
        const data = await fetchIndicators(sym);
        removeTyping(typingEl);
        const latest = data.indicators[data.indicators.length - 1];
        const sentiment = data.overall_sentiment_score;

        // Set active stock context globally (Fix for Item #3)
        activeStockSymbol = sym;

        // Use deterministic technical bias calculated by backend (Fix for Item #4)
        const bias = latest.overall_technical_bias || getBias(latest);
        const rsiSignal = latest.rsi > 70 ? 'overbought' : latest.rsi < 30 ? 'oversold' : 'neutral momentum';
        const macdSignal = (latest.macd > latest.macd_signal) ? 'bullish crossover' : 'bearish crossover';
        const vwapSignal = latest.price > latest.vwap ? 'trading above VWAP — intraday buying pressure' : 'trading below VWAP — selling pressure dominant';

        const narrative = `${formatSymbolName(sym)} is currently trading at ₹${latest.price.toLocaleString('en-IN', {minimumFractionDigits: 2})}. The 5 EMA stands at ₹${latest.ema_5?.toFixed(2)}, suggesting the stock is ${latest.price > latest.ema_5 ? 'above its short-term trend — a bullish signal' : 'below its short-term trend — bearish near-term'}. RSI at ${latest.rsi?.toFixed(1)} indicates ${rsiSignal}. MACD shows a ${macdSignal}, and the stock is ${vwapSignal}. Overall technical bias is <strong style="color:${bias==='BULLISH'?'var(--green)':bias==='BEARISH'?'var(--red)':'var(--accent)'}">${bias}</strong>.`;

        const statCard = buildStatCard([
            { label: 'Price', value: `₹${latest.price?.toFixed(2)}`, cls: '' },
            { label: '5 EMA', value: `₹${latest.ema_5?.toFixed(2)}`, cls: '' },
            { label: 'RSI', value: latest.rsi?.toFixed(1), cls: latest.rsi > 70 ? 'bearish' : latest.rsi < 30 ? 'bullish' : '' },
            { label: 'MACD', value: latest.macd?.toFixed(3), cls: latest.macd > 0 ? 'bullish' : 'bearish' },
            { label: 'VWAP', value: `₹${latest.vwap?.toFixed(2)}`, cls: '' },
            { label: 'Sentiment', value: sentiment >= 0 ? `+${sentiment.toFixed(3)}` : sentiment.toFixed(3), cls: sentiment >= 0 ? 'bullish' : 'bearish' },
        ]);

        const actions = `<div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;">
            <button class="btn-ghost" style="font-size:0.78rem;" onclick="requestCharts('${sym}')">View technical charts</button>
            <button class="btn-ghost" style="font-size:0.78rem;border-color:var(--accent);color:var(--accent);" onclick="requestInvest('${sym}')">Allocate capital to this stock</button>
        </div>`;

        appendAIMessage(narrative, statCard + actions);

        // Store chart data for later
        currentChartsData[sym] = data;
        addToChartSelector(sym);

        const symName = formatSymbolName(sym);
        updateSuggestionChips([
            `View ${symName} charts`,
            `Compare ${symName} sector peers`,
            `Allocate ₹50,000 to ${symName}`,
            `Explain 5 EMA for ${symName}`
        ]);

    } catch (err) {
        removeTyping(typingEl);
        appendAIMessage(`I wasn't able to fetch live data for ${formatSymbolName(sym)} right now. This could be a network issue or the ticker may not be available via yfinance. Please try again in a moment.`, null);
    }
}

async function handleChartIntent(intent, typingEl) {
    // Fall back to active stock context if no stock named in user prompt (Fix for Item #3)
    const sym = intent.symbol || activeStockSymbol;
    if (!sym) {
        removeTyping(typingEl);
        appendAIMessage('Which stock would you like to see charts for? You can say something like "show me charts for Reliance" or "HDFCBANK technical chart".', null);
        return;
    }
    removeTyping(typingEl);
    requestCharts(sym);
    appendAIMessage(`Loading technical charts for ${formatSymbolName(sym)}. You can switch between the Price / EMA, RSI, and MACD tabs below.`, null);
    updateSuggestionChips([
        'Explain 5 EMA vs VWAP',
        'Explain RSI momentum',
        'Explain MACD crossover',
        'Allocate capital to this stock'
    ]);
}


async function handleInvestIntent(intent, typingEl) {
    removeTyping(typingEl);
    const amount = intent.amount;

    if (amount) {
        document.getElementById('capital-input').value = amount;
        appendAIMessage(`Understood — allocating ₹${amount.toLocaleString('en-IN')}. Here is the investment panel below. Review the settings and click "Scan Market" when you're ready.`, null);
    } else {
        appendAIMessage(`Ready to allocate capital. How much would you like to invest? You can type an amount like "₹50,000" or enter it directly in the panel below.`, null);
    }

    revealSection('invest-section');
    revealSection('peers-section');
    document.getElementById('invest-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    updateSuggestionChips([
        'Scan Market and Generate Recommendations',
        'Which banking stocks look strong?',
        'Tell me about HDFC Bank'
    ]);
}

async function handleSectorIntent(intent, typingEl) {
    const sector = intent.sector;
    const sectorStocks = {
        banking: ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS'],
        it:      ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'],
        auto:    ['TATAMOTORS.NS', 'MARUTI.NS'],
        pharma:  ['SUNPHARMA.NS'],
        fmcg:    ['ITC.NS', 'TITAN.NS', 'ASIANPAINT.NS'],
        energy:  ['RELIANCE.NS'],
    };
    const stocks = sectorStocks[sector] || [];
    removeTyping(typingEl);

    if (!stocks.length) {
        appendAIMessage(`I don't have a predefined stock list for that sector. Try asking about a specific stock like "Tell me about Reliance".`, null);
        return;
    }

    appendAIMessage(
        `The ${sector.toUpperCase()} sector on NSE includes: <strong>${stocks.map(formatSymbolName).join(', ')}</strong>. I can pull live data for any of these — just ask about one specifically, or tell me you want to invest and I'll scan the full sector.`,
        `<div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;">
            ${stocks.map(s => `<button class="chip" onclick="sendSuggestion('Tell me about ${formatSymbolName(s)}')">${formatSymbolName(s)}</button>`).join('')}
        </div>`
    );

    const s1 = stocks[0] ? formatSymbolName(stocks[0]) : 'HDFC Bank';
    const s2 = stocks[1] ? formatSymbolName(stocks[1]) : 'Reliance';
    updateSuggestionChips([
        `Tell me about ${s1}`,
        `Tell me about ${s2}`,
        `Compare ${sector.toUpperCase()} sector peers`,
        'I want to invest ₹50,000'
    ]);
}

async function handleMarketIntent(typingEl) {
    removeTyping(typingEl);
    const hour = new Date().getHours();
    const marketOpen = hour >= 9 && hour < 16;
    const status = marketOpen ? 'NSE is currently within trading hours (9:15 AM – 3:30 PM IST)' : 'NSE markets are closed for regular trading sessions';

    appendAIMessage(
        `${status}. I can analyse live price data, technical indicators, and sentiment for any of the top 20 NSE liquid equities in my universe — covering IT, Banking, Auto, FMCG, Pharma, and Energy sectors. Ask me about a specific stock or sector to get started, or tell me your investment amount to generate a full intraday allocation plan.`,
        `<div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;">
            <button class="chip" onclick="sendSuggestion('Tell me about HDFC Bank')">HDFC Bank</button>
            <button class="chip" onclick="sendSuggestion('Tell me about Reliance')">Reliance</button>
            <button class="chip" onclick="sendSuggestion('Tell me about TCS')">TCS</button>
            <button class="chip" onclick="sendSuggestion('Which banking stocks look strong?')">Banking sector</button>
        </div>`
    );
    updateSuggestionChips([
        'Tell me about HDFC Bank',
        'Tell me about Reliance',
        'Compare IT sector stocks',
        'I want to invest ₹50,000'
    ]);
}

async function handleGeneralIntent(msg, typingEl) {
    // Fall back to backend chat with full conversation transcript (Fix for Item #1)
    try {
        const r = await fetch('/api/v1/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                session_id: 'default_session',
                history: chatHistory
            })
        });
        const data = await r.json();
        removeTyping(typingEl);
        appendAIMessage(data.answer || 'I can help you with NSE stock analysis, market conditions, and intraday capital allocation. Try asking about a specific stock or sector.', null, data.provider_used);
    } catch {
        removeTyping(typingEl);
        appendAIMessage('I can help you with NSE stock analysis, market conditions, and intraday capital allocation. Try asking about a specific stock or sector.', null);
    }
    updateSuggestionChips([
        'Tell me about HDFC Bank',
        'Tell me about Reliance',
        'Compare IT sector stocks',
        'I want to invest ₹50,000'
    ]);
}


// ── Triggered by chat actions ──────────────────────────────────────────────
async function requestCharts(symbol) {
    revealSection('charts-section');

    if (!currentChartsData[symbol]) {
        try {
            const data = await fetchIndicators(symbol);
            currentChartsData[symbol] = data;
            addToChartSelector(symbol);
        } catch { return; }
    }

    const sel = document.getElementById('stock-chart-selector');
    sel.value = symbol;
    renderCharts(symbol);
    document.getElementById('charts-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function requestInvest(symbol) {
    revealSection('invest-section');
    revealSection('peers-section');
    document.getElementById('invest-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Market Scan ────────────────────────────────────────────────────────────
async function handleScanMarket() {
    const btn = document.getElementById('scan-btn');
    const capital = parseFloat(document.getElementById('capital-input').value);

    if (!capital || capital < 500) {
        alert('Please enter a valid investment amount (minimum ₹500).');
        return;
    }

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="margin-right:8px;"></span> Scanning market...`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25000);

    try {
        const r = await fetch('/api/v1/recommendations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({ investment_amount: capital })
        });
        clearTimeout(timeoutId);

        if (!r.ok) {
            const err = await r.json();
            throw new Error(err.detail || `HTTP ${r.status}`);
        }

        const data = await r.json();
        currentRecommendations = data.recommendations || [];

        // Merge chart data from scan
        if (data.charts) {
            Object.assign(currentChartsData, data.charts);
            for (const sym of Object.keys(data.charts)) addToChartSelector(sym);
        }

        renderRecommendations(data);
        populatePeerSelector(currentRecommendations);
        revealSection('recommendations-section');
        revealSection('charts-section');

        // AI message summary
        const topPick = currentRecommendations[0];
        if (topPick) {
            appendAIMessage(
                `Scan complete. I found <strong>${currentRecommendations.length} high-confidence trade setups</strong> across your ₹${capital.toLocaleString('en-IN')} allocation. The top pick is <strong>${formatSymbolName(topPick.symbol)}</strong> — ${topPick.action} signal at ${topPick.confidence}% confidence, with ₹${topPick.allocated_capital?.toFixed(2)} allocated. Scroll down for the full breakdown, charts, and sector peer comparison.`,
                null
            );
        }
        updateSuggestionChips([
            'Compare sector peers',
            'Explain top pick rationale',
            'View technical charts'
        ]);

        document.getElementById('recommendations-section').scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
        clearTimeout(timeoutId);
        const msg = err.name === 'AbortError'
            ? 'The market scan timed out. NSE data may be slow — please try again.'
            : `Scan failed: ${err.message}`;
        appendAIMessage(msg, null);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Scan Market and Generate Recommendations';
    }
}

// ── Render Recommendations ─────────────────────────────────────────────────
function renderRecommendations(data) {
    const allocated = Math.min(data.investment_amount, data.total_allocated);
    const unallocated = Math.max(0, data.investment_amount - allocated);

    document.getElementById('stat-total').textContent = `₹${data.investment_amount?.toLocaleString('en-IN')}`;
    document.getElementById('stat-allocated').textContent = `₹${allocated.toFixed(2)}`;
    document.getElementById('stat-unallocated').textContent = `₹${unallocated.toFixed(2)}`;

    if (data.trace) {
        document.getElementById('trace-latency').textContent = `${data.trace.execution_time_seconds}s`;
        document.getElementById('trace-provider').textContent = data.trace.provider_used;
        document.getElementById('trace-tools').textContent = data.trace.tool_calls_count;
        document.getElementById('trace-fallback').textContent = data.trace.fallbacks_triggered?.length ? data.trace.fallbacks_triggered.join(', ') : 'None';
    }

    // Recommendation cards
    const container = document.getElementById('recommendations-container');
    container.innerHTML = '';

    (data.recommendations || []).forEach(rec => {
        const isBuy = rec.action === 'BUY';
        const card = document.createElement('div');
        card.className = `rec-card ${isBuy ? 'buy-card' : 'sell-card'}`;
        const confPct = rec.confidence || 0;
        card.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;position:relative;z-index:1;">
                <div>
                    <div style="font-size:0.72rem;font-weight:600;color:var(--text-subtle);letter-spacing:0.04em;text-transform:uppercase;margin-bottom:4px;">${rec.symbol}</div>
                    <div style="font-size:1.3rem;font-weight:700;letter-spacing:-0.02em;color:var(--text);">₹${rec.current_price?.toFixed(2)}</div>
                </div>
                <span class="badge ${isBuy ? 'badge-buy' : 'badge-sell'}">${rec.action}</span>
            </div>
            <div style="margin-bottom:14px;position:relative;z-index:1;">
                <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:var(--text-subtle);margin-bottom:5px;">
                    <span>Model Confidence</span>
                    <span style="font-weight:600;color:${isBuy?'var(--green)':'var(--red)'};">${confPct}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:${confPct}%;background:${isBuy?'var(--green)':'var(--red)'};"></div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.78rem;margin-bottom:14px;position:relative;z-index:1;">
                <div>
                    <div style="color:var(--text-subtle);font-size:0.68rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">Allocated</div>
                    <div style="font-weight:600;color:var(--text);">₹${rec.allocated_capital?.toFixed(2)}</div>
                </div>
                <div>
                    <div style="color:var(--text-subtle);font-size:0.68rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">Shares</div>
                    <div style="font-weight:600;color:var(--text);">${rec.shares_to_trade}</div>
                </div>
                <div>
                    <div style="color:var(--text-subtle);font-size:0.68rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">Target</div>
                    <div style="font-weight:600;color:var(--green);">₹${rec.target_price?.toFixed(2)}</div>
                </div>
                <div>
                    <div style="color:var(--text-subtle);font-size:0.68rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">Stop Loss</div>
                    <div style="font-weight:600;color:var(--red);">₹${rec.stop_loss?.toFixed(2)}</div>
                </div>
            </div>
            ${rec.rationale ? `<div style="font-size:0.75rem;color:var(--text-muted);line-height:1.55;padding-top:12px;border-top:1px solid var(--border);position:relative;z-index:1;">${rec.rationale}</div>` : ''}
            <div style="margin-top:12px;position:relative;z-index:1;">
                <button class="btn-ghost" style="font-size:0.75rem;width:100%;text-align:center;" onclick="requestCharts('${rec.symbol}')">View charts</button>
            </div>
        `;
        container.appendChild(card);
    });

    // Doughnut chart
    renderAllocationChart(data.recommendations || [], unallocated);
}

function renderAllocationChart(recs, unallocated) {
    const labels = [...recs.map(r => r.symbol.replace('.NS', '')), 'Cash Reserve'];
    const values = [...recs.map(r => r.allocated_capital || 0), unallocated];
    const colors = [
        'rgba(249,115,22,0.85)',
        'rgba(34,197,94,0.85)',
        'rgba(249,115,22,0.55)',
        'rgba(34,197,94,0.55)',
        'rgba(163,163,163,0.4)',
    ];

    if (allocationChartInstance) { allocationChartInstance.destroy(); }
    const ctx = document.getElementById('allocationChart').getContext('2d');
    allocationChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{ data: values, backgroundColor: colors.slice(0, values.length), borderWidth: 0, hoverOffset: 6 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, cutout: '70%',
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ` ₹${c.raw.toFixed(2)}` } } }
        }
    });

    const legend = document.getElementById('allocation-legend');
    legend.innerHTML = labels.map((l, i) => `
        <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.8rem;">
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:10px;height:10px;border-radius:3px;background:${colors[i] || '#999'};"></div>
                <span style="color:var(--text-muted);">${l}</span>
            </div>
            <span style="font-weight:600;color:var(--text);">₹${values[i].toFixed(2)}</span>
        </div>
    `).join('');
}

// ── Charts ─────────────────────────────────────────────────────────────────
function initCharts() {
    const opts = (yLabel) => ({
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: cssVar('--text-muted'), font: { size: 10, family: 'inherit' }, boxWidth: 12 } } },
        scales: {
            x: { ticks: { color: cssVar('--text-subtle'), font: { size: 9 } }, grid: { color: cssVar('--border') } },
            y: { ticks: { color: cssVar('--text-subtle'), font: { size: 9 } }, grid: { color: cssVar('--border') } }
        }
    });
    priceEmaChartInstance = new Chart(document.getElementById('priceEmaChart').getContext('2d'), { type: 'line', data: { labels: [], datasets: [] }, options: opts('Price') });
    rsiChartInstance      = new Chart(document.getElementById('rsiChart').getContext('2d'),     { type: 'line', data: { labels: [], datasets: [] }, options: opts('RSI') });
    macdChartInstance     = new Chart(document.getElementById('macdChart').getContext('2d'),    { type: 'line', data: { labels: [], datasets: [] }, options: opts('MACD') });
}

function addToChartSelector(symbol) {
    const sel = document.getElementById('stock-chart-selector');
    if ([...sel.options].some(o => o.value === symbol)) return;
    const opt = document.createElement('option');
    opt.value = symbol;
    opt.textContent = symbol.replace('.NS', '');
    sel.appendChild(opt);
}

function handleStockChartSelection() {
    const sym = document.getElementById('stock-chart-selector').value;
    if (sym && currentChartsData[sym]) renderCharts(sym);
}

function switchTab(btn, tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.chart-tab').forEach(c => c.style.display = 'none');
    document.getElementById(`tab-${tab}`).style.display = 'flex';
}

function renderCharts(symbol) {
    const data = currentChartsData[symbol];
    if (!data || !data.indicators) return;
    const pts = data.indicators;
    const labels = pts.map(p => p.date?.slice(5));
    const accent = cssVar('--accent');
    const green = cssVar('--green');
    const muted = cssVar('--text-subtle');

    // Price & EMA
    priceEmaChartInstance.data.labels = labels;
    priceEmaChartInstance.data.datasets = [
        { label: 'Price', data: pts.map(p => p.price), borderColor: accent, borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false },
        { label: '5 EMA', data: pts.map(p => p.ema_5), borderColor: green, borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false, borderDash: [4, 3] },
        { label: 'VWAP', data: pts.map(p => p.vwap), borderColor: muted, borderWidth: 1, pointRadius: 0, tension: 0.3, fill: false, borderDash: [2, 4] },
    ];
    priceEmaChartInstance.update();

    // RSI
    rsiChartInstance.data.labels = labels;
    rsiChartInstance.data.datasets = [
        { label: 'RSI', data: pts.map(p => p.rsi), borderColor: accent, borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false },
    ];
    rsiChartInstance.options.plugins.annotation = {};
    rsiChartInstance.update();

    // MACD
    macdChartInstance.data.labels = labels;
    macdChartInstance.data.datasets = [
        { label: 'MACD', data: pts.map(p => p.macd), borderColor: accent, borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false },
        { label: 'Signal', data: pts.map(p => p.macd_signal), borderColor: green, borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false, borderDash: [4, 3] },
    ];
    macdChartInstance.update();

    // Dynamic explanations
    const latest = pts[pts.length - 1];
    const bias = latest?.price > latest?.ema_5 ? 'trading above its 5 EMA, confirming bullish short-term momentum.' : 'trading below its 5 EMA — near-term momentum is bearish.';
    document.getElementById('price-explanation-dynamic').textContent = `${symbol.replace('.NS', '')} is currently ${bias} VWAP: ₹${latest?.vwap?.toFixed(2)}.`;

    const rsiVal = latest?.rsi?.toFixed(1);
    const rsiTxt = latest?.rsi > 70 ? `RSI at ${rsiVal} — overbought. Consider waiting for a pullback before entering.` : latest?.rsi < 30 ? `RSI at ${rsiVal} — oversold. A bounce could be imminent.` : `RSI at ${rsiVal} — neutral. No extreme momentum signals.`;
    document.getElementById('rsi-explanation-dynamic').textContent = rsiTxt;

    const macdTxt = latest?.macd > latest?.macd_signal ? `MACD line is above the Signal line — bullish momentum. A potential entry signal.` : `MACD line is below the Signal line — bearish crossover. Exercise caution.`;
    document.getElementById('macd-explanation-dynamic').textContent = macdTxt;

    updateChartColors();
}

function updateChartColors() {
    [priceEmaChartInstance, rsiChartInstance, macdChartInstance].forEach(c => {
        if (!c) return;
        c.options.scales.x.ticks.color = cssVar('--text-subtle');
        c.options.scales.x.grid.color  = cssVar('--border');
        c.options.scales.y.ticks.color = cssVar('--text-subtle');
        c.options.scales.y.grid.color  = cssVar('--border');
        if (c.options.plugins?.legend?.labels) c.options.plugins.legend.labels.color = cssVar('--text-muted');
        c.update();
    });
}

// ── Sector Peers ───────────────────────────────────────────────────────────
function populatePeerSelector(recs) {
    const sel = document.getElementById('peer-stock-selector');
    sel.innerHTML = '';
    if (!recs.length) { sel.innerHTML = '<option value="">Run a scan first</option>'; return; }
    recs.forEach(r => {
        const opt = document.createElement('option');
        opt.value = r.symbol;
        opt.textContent = r.symbol.replace('.NS', '') + ` — ${r.action}`;
        sel.appendChild(opt);
    });
}

async function handlePeerCompare() {
    const symbol = document.getElementById('peer-stock-selector').value;
    const maxPeers = parseInt(document.getElementById('peer-max-count').value, 10);
    const btn = document.getElementById('peer-compare-btn');
    const statusEl = document.getElementById('peers-status');
    const tableWrap = document.getElementById('peers-table-wrap');
    const chartWrap = document.getElementById('peers-chart-wrap');

    if (!symbol) { alert('Please run a market scan first, then select a stock.'); return; }

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="margin-right:6px;"></span> Fetching peers...`;
    statusEl.textContent = `Querying Chroma vector DB for ${symbol.replace('.NS','')} sector peers...`;
    statusEl.classList.remove('hidden');
    tableWrap.classList.add('hidden');
    chartWrap.classList.add('hidden');

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 35000);

    try {
        const r = await fetch('/api/v1/peers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({ symbol, max_peers: maxPeers })
        });
        clearTimeout(timeoutId);
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || `HTTP ${r.status}`); }
        const peers = await r.json();
        if (!peers?.length) { statusEl.textContent = `No sector peers found for ${symbol}. Run a scan first to index fingerprints.`; return; }
        statusEl.textContent = `Found ${peers.length - 1} peer(s) in the ${peers[0].sector} sector.`;
        renderPeerTable(peers);
        renderPeersChart(peers);
    } catch (err) {
        clearTimeout(timeoutId);
        statusEl.textContent = err.name === 'AbortError' ? 'Peer fetch timed out — try again.' : `Error: ${err.message}`;
        statusEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Find Sector Peers';
    }
}

function renderPeerTable(rows) {
    const tbody = document.getElementById('peers-table-body');
    tbody.innerHTML = '';
    rows.forEach(row => {
        const isOrig = row.is_original;
        const rsiCls = row.rsi > 70 ? 'color:var(--red)' : row.rsi < 30 ? 'color:var(--green)' : 'color:var(--text-muted)';
        const sentCls = row.sentiment_score >= 0 ? 'color:var(--green)' : 'color:var(--red)';
        const dirCls = row.action === 'BUY' ? 'color:var(--green);font-weight:700' : 'color:var(--red);font-weight:700';
        const simPct = Math.round((row.similarity_score || 0) * 100);
        const tr = document.createElement('tr');
        if (isOrig) tr.classList.add('highlight-row');

        const hasValidPrice = row.current_price && row.current_price > 0;
        const priceStr = hasValidPrice ? `₹${row.current_price.toFixed(2)}` : `<span style="color:var(--red);font-size:0.75rem;">Data Unavailable</span>`;
        const emaStr   = hasValidPrice && row.ema_5 ? `₹${row.ema_5.toFixed(2)}` : '—';
        const vwapStr  = hasValidPrice && row.vwap ? `₹${row.vwap.toFixed(2)}` : '—';
        const rsiStr   = hasValidPrice && row.rsi !== undefined ? row.rsi.toFixed(1) : '—';
        const macdStr  = hasValidPrice && row.macd !== undefined ? row.macd.toFixed(3) : '—';

        tr.innerHTML = `
            <td style="font-weight:700;color:var(--text);">${row.symbol.replace('.NS','')}${isOrig ? `<span style="margin-left:8px;font-size:0.68rem;background:var(--accent-muted);color:var(--accent);padding:2px 8px;border-radius:999px;font-weight:600;">ORIGINAL</span>` : ''}</td>
            <td style="text-align:center;">
                <div style="display:flex;align-items:center;gap:6px;justify-content:center;">
                    <div style="width:50px;height:4px;background:var(--border);border-radius:999px;overflow:hidden;">
                        <div style="width:${simPct}%;height:100%;background:var(--accent);border-radius:999px;"></div>
                    </div>
                    <span style="font-size:0.78rem;font-weight:600;color:var(--accent);">${simPct}%</span>
                </div>
            </td>
            <td style="text-align:center;font-weight:600;">${priceStr}</td>
            <td style="text-align:center;color:var(--text-muted);">${emaStr}</td>
            <td style="text-align:center;font-weight:700;${hasValidPrice ? rsiCls : ''}">${rsiStr}</td>
            <td style="text-align:center;color:var(--text-muted);">${macdStr}</td>
            <td style="text-align:center;color:var(--text-muted);">${vwapStr}</td>
            <td style="text-align:center;font-weight:600;${sentCls}">${row.sentiment_score>=0?'+':''}${(row.sentiment_score||0).toFixed(3)}</td>
            <td style="text-align:center;font-weight:700;color:var(--accent);">${hasValidPrice ? (row.confidence||0)+'%' : '—'}</td>
            <td style="text-align:center;${hasValidPrice ? dirCls : ''}">${hasValidPrice ? (row.action||'—') : '—'}</td>
        `;
        tbody.appendChild(tr);
    });
    document.getElementById('peers-table-wrap').classList.remove('hidden');
}

function renderPeersChart(rows) {
    const sorted = [...rows].sort((a,b) => (b.confidence||0) - (a.confidence||0));
    const labels = sorted.map(r => r.symbol.replace('.NS',''));
    const confidences = sorted.map(r => r.confidence||0);
    const bgColors = sorted.map(r => r.is_original ? 'rgba(249,115,22,0.9)' : 'rgba(249,115,22,0.4)');

    document.getElementById('peers-chart-wrap').classList.remove('hidden');
    if (peersChartInstance) { peersChartInstance.destroy(); peersChartInstance = null; }

    peersChartInstance = new Chart(document.getElementById('peersConfidenceChart').getContext('2d'), {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Confidence %', data: confidences, backgroundColor: bgColors, borderWidth: 0, borderRadius: 6 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ` ${c.parsed.y}% confidence` } } },
            scales: {
                x: { ticks: { color: cssVar('--text-subtle'), font: { size: 10 } }, grid: { color: cssVar('--border') } },
                y: { min: 50, max: 100, ticks: { color: cssVar('--text-subtle'), font: { size: 10 }, callback: v => v + '%' }, grid: { color: cssVar('--border') } }
            }
        }
    });
}

// ── Chat Message Rendering ─────────────────────────────────────────────────
function appendUserMessage(text) {
    chatHistory.push({ role: 'user', content: text });
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'bubble-user';
    div.style.alignSelf = 'flex-end';
    div.textContent = text;
    container.appendChild(div);
    scrollChat();
}

function appendAIMessage(htmlText, extraHtml, providerUsed) {
    chatHistory.push({ role: 'agent', content: htmlText.replace(/<[^>]*>/g, '') });
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'bubble-ai';

    let badgeHtml = '';
    if (providerUsed) {
        const nameMap = { 'gemini': 'Google Gemini', 'groq': 'Groq Llama 3', 'openrouter': 'OpenRouter', 'offline_knowledge_base': 'Knowledge Base' };
        const label = nameMap[providerUsed] || providerUsed;
        badgeHtml = `<div style="font-size:0.68rem;color:var(--text-subtle);margin-top:8px;display:flex;align-items:center;gap:4px;justify-content:flex-end;">
            <span style="width:5px;height:5px;border-radius:50%;background:var(--accent);"></span>
            <span>Handled by ${label}</span>
        </div>`;
    }

    div.innerHTML = `<div style="font-size:0.85rem;color:var(--text);line-height:1.65;">${htmlText}</div>${extraHtml ? `<div style="margin-top:0;">${extraHtml}</div>` : ''}${badgeHtml}`;
    container.appendChild(div);
    scrollChat();
}


function showTyping() {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'bubble-ai';
    div.id = 'typing-indicator';
    div.innerHTML = `<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
    container.appendChild(div);
    scrollChat();
    return div;
}

function removeTyping(el) { if (el && el.parentNode) el.remove(); }

function scrollChat() {
    const c = document.getElementById('chat-messages');
    c.scrollTop = c.scrollHeight;
}

function hideSuggestions() {
    const s = document.getElementById('suggestions-bar');
    if (s) s.style.display = 'none';
}

// ── Section Reveal ─────────────────────────────────────────────────────────
function revealSection(id) {
    const el = document.getElementById(id);
    if (!el || !el.classList.contains('hidden')) return;
    el.classList.remove('hidden');
    el.classList.add('section-reveal');
}

// ── Helpers ────────────────────────────────────────────────────────────────
async function fetchIndicators(symbol) {
    const r = await fetch(`/api/v1/indicators/${encodeURIComponent(symbol)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
}

function formatSymbolName(sym) {
    const names = {
        'RELIANCE.NS':'Reliance', 'TCS.NS':'TCS', 'INFY.NS':'Infosys',
        'HDFCBANK.NS':'HDFC Bank', 'ICICIBANK.NS':'ICICI Bank', 'TATAMOTORS.NS':'Tata Motors',
        'BHARTIARTL.NS':'Airtel', 'SBIN.NS':'SBI', 'AXISBANK.NS':'Axis Bank',
        'ITC.NS':'ITC', 'WIPRO.NS':'Wipro', 'BAJFINANCE.NS':'Bajaj Finance',
        'MARUTI.NS':'Maruti', 'LT.NS':'L&T', 'HCLTECH.NS':'HCL Tech',
        'SUNPHARMA.NS':'Sun Pharma', 'TITAN.NS':'Titan', 'ULTRACEMCO.NS':'UltraTech',
        'ASIANPAINT.NS':'Asian Paints', 'KOTAKBANK.NS':'Kotak Bank',
    };
    return names[sym] || sym.replace('.NS','');
}

function getBias(latest) {
    const p = latest?.price || latest?.close_price || 0;
    const emaScore  = p > (latest?.ema_5 || 0) ? 1 : -1;
    const vwapScore = p > (latest?.vwap || 0) ? 1 : -1;
    const macdScore = (latest?.macd || 0) > (latest?.macd_signal || 0) ? 1 : -1;
    const rsiScore  = (latest?.rsi || 50) >= 50 ? 1 : -1;

    const total = emaScore + vwapScore + macdScore + rsiScore;
    if (total >= 2)  return 'BULLISH';
    if (total <= -2) return 'BEARISH';
    return 'NEUTRAL';
}

function buildStatCard(stats) {
    return `<div class="stat-card">${stats.map(s => `
        <div class="stat-cell">
            <span class="stat-label">${s.label}</span>
            <span class="stat-value ${s.cls}">${s.value}</span>
        </div>`).join('')}
    </div>`;
}

function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
