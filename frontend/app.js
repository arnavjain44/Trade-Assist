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



// ── Chat Input Helper ──────────────────────────────────────────────────────
function setChatInputEnabled(enabled) {
    const input = document.getElementById('chat-input');
    const btn = document.getElementById('send-btn');
    if (input) {
        input.disabled = !enabled;
        input.readOnly = false;
        if (enabled) {
            try { input.focus(); } catch (e) {}
        }
    }
    if (btn) {
        btn.disabled = !enabled;
    }
}

// ── Chat Send ──────────────────────────────────────────────────────────────
async function handleChatSend() {
    const input = document.getElementById('chat-input');
    const msg = input ? input.value.trim() : '';
    if (!msg) return;

    hideSuggestions();
    appendUserMessage(msg);
    if (input) input.value = '';

    setChatInputEnabled(false);
    const typingEl = showTyping();
    const intent = detectIntent(msg);

    try {
        switch (intent.type) {
            case 'stock':   await handleStockIntent(intent, typingEl, msg); break;
            case 'chart':   await handleChartIntent(intent, typingEl, msg); break;
            case 'invest':  await handleInvestIntent(intent, typingEl, msg); break;
            case 'sector':  await handleSectorIntent(intent, typingEl, msg); break;
            case 'market':  await handleMarketIntent(typingEl, msg); break;
            default:        await handleGeneralIntent(msg, typingEl); break;
        }
    } catch (err) {
        removeTyping(typingEl);
        appendAIMessage("I couldn't reach the AI service right now. Your market-data analysis is still separate from the chat service.", null);
    } finally {
        setChatInputEnabled(true);
    }
}

function sendSuggestion(text) {
    const input = document.getElementById('chat-input');
    if (input) input.value = text;
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
async function handleStockIntent(intent, typingEl, userPrompt) {
    const sym = intent.symbol;
    if (!sym) {
        return await handleGeneralIntent(userPrompt, typingEl);
    }
    try {
        const data = await fetchIndicators(sym);
        const latest = (data.indicators && data.indicators.length > 0) ? data.indicators[data.indicators.length - 1] : {};
        const pred = data.latest_prediction || {};
        const sentiment = data.overall_sentiment_score || 0;
        const ctxFeats = pred.context_features || {};

        activeStockSymbol = sym;

        const pLongPct = (pred.p_long !== undefined && pred.p_long !== null) ? (pred.p_long * 100).toFixed(2) : 'N/A';
        const pShortPct = (pred.p_short !== undefined && pred.p_short !== null) ? (pred.p_short * 100).toFixed(2) : 'N/A';
        const threshPct = (pred.model_threshold !== undefined && pred.model_threshold !== null) ? (pred.model_threshold * 100).toFixed(2) : '80.00';
        const action = pred.action || 'HOLD';
        const modelName = pred.model_name || 'phase5_d_lightgbm';

        const stockAnalysisContext = {
            symbol: sym,
            current_price: latest.price || pred.current_price,
            ema_5: latest.ema_5,
            rsi: latest.rsi,
            macd: latest.macd,
            macd_signal: latest.macd_signal,
            vwap: latest.vwap,
            overall_sentiment_score: sentiment,
            p_long: pred.p_long,
            p_short: pred.p_short,
            threshold: pred.model_threshold || 0.8000,
            action: action,
            qualified: pred.qualified || false,
            market_similarity: ctxFeats.market_similarity || 0.0,
            stock_similarity: ctxFeats.stock_similarity || 0.0,
        };

        let llmExplanation = '';
        let providerUsed = 'fallback';
        try {
            const chatRes = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userPrompt || `Explain the technical and ML analysis for ${sym}.`,
                    session_id: 'default_session',
                    history: chatHistory,
                    stock_analysis: stockAnalysisContext
                })
            });
            if (chatRes.ok) {
                const chatData = await chatRes.json();
                llmExplanation = chatData.answer || '';
                providerUsed = chatData.provider_used || 'fallback';
            } else {
                llmExplanation = "I couldn't reach the AI explanation service right now. Your market-data analysis is shown below.";
            }
        } catch (e) {
            console.warn('Failed to fetch LLM explanation:', e);
            llmExplanation = "I couldn't reach the AI explanation service right now. Your market-data analysis is shown below.";
        }

        removeTyping(typingEl);

        const modelHeader = `<div style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:12px;padding:12px 16px;margin-bottom:14px;">
            <div style="font-size:0.75rem;font-weight:600;color:var(--text-subtle);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">Model: ${modelName}</div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(110px, 1fr));gap:8px;font-size:0.8rem;margin-bottom:6px;">
                <div><span style="color:var(--text-subtle);">LONG prob:</span> <strong>${pLongPct}%</strong></div>
                <div><span style="color:var(--text-subtle);">SHORT prob:</span> <strong>${pShortPct}%</strong></div>
                <div><span style="color:var(--text-subtle);">Threshold:</span> <strong>${threshPct}%</strong></div>
                <div><span style="color:var(--text-subtle);">Decision:</span> <strong style="color:${action==='BUY'?'var(--green)':action==='SELL'?'var(--red)':'var(--accent)'}">${action}</strong></div>
            </div>
        </div>`;

        const statCard = buildStatCard([
            { label: 'Price', value: latest.price ? `₹${latest.price.toFixed(2)}` : 'N/A', cls: '' },
            { label: '5 EMA', value: latest.ema_5 ? `₹${latest.ema_5.toFixed(2)}` : 'N/A', cls: '' },
            { label: 'RSI', value: latest.rsi ? latest.rsi.toFixed(1) : 'N/A', cls: latest.rsi > 70 ? 'bearish' : latest.rsi < 30 ? 'bullish' : '' },
            { label: 'MACD', value: latest.macd ? latest.macd.toFixed(3) : 'N/A', cls: latest.macd > 0 ? 'bullish' : 'bearish' },
            { label: 'VWAP', value: latest.vwap ? `₹${latest.vwap.toFixed(2)}` : 'N/A', cls: '' },
            { label: 'Sentiment', value: sentiment >= 0 ? `+${sentiment.toFixed(3)}` : sentiment.toFixed(3), cls: sentiment >= 0 ? 'bullish' : 'bearish' },
        ]);

        const actions = `<div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;">
            <button class="btn-ghost" style="font-size:0.78rem;" onclick="requestCharts('${sym}')">View technical charts</button>
            <button class="btn-ghost" style="font-size:0.78rem;border-color:var(--accent);color:var(--accent);" onclick="requestInvest('${sym}')">Use this stock in a market scan</button>
        </div>`;

        const fullMessage = `${modelHeader}${statCard}<div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--border);"><div style="font-size:0.75rem;font-weight:600;color:var(--text-subtle);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">AI Explanation</div><div style="line-height:1.6;">${llmExplanation}</div></div>${actions}`;

        appendAIMessage(fullMessage, null, providerUsed);

        currentChartsData[sym] = data;
        addToChartSelector(sym);

        const symName = formatSymbolName(sym);
        updateSuggestionChips([
            `View ${symName} charts`,
            `Compare ${symName} sector peers`,
            `Allocate ₹50,000 to ${symName}`,
            `Explain 5 EMA for ${symName}`
        ]);

        if (intent.amount || (userPrompt && userPrompt.toLowerCase().includes('invest'))) {
            if (intent.amount) {
                document.getElementById('capital-input').value = intent.amount;
            }
            revealSection('invest-section');
            revealSection('peers-section');
        }

    } catch (err) {
        removeTyping(typingEl);
        appendAIMessage(`I wasn't able to fetch live data for ${formatSymbolName(sym)} right now. This could be a network issue or the ticker may not be available via yfinance. Please try again in a moment.`, null);
    }
}


async function handleChartIntent(intent, typingEl, userPrompt) {
    const sym = intent.symbol || activeStockSymbol;
    if (!sym) {
        removeTyping(typingEl);
        appendAIMessage('Which stock would you like to see charts for? You can say something like "show me charts for Reliance" or "HDFCBANK technical chart".', null);
        return;
    }
    revealSection('charts-section');
    requestCharts(sym);

    let llmExplanation = '';
    let providerUsed = null;
    try {
        const data = currentChartsData[sym] || await fetchIndicators(sym);
        const latest = (data.indicators && data.indicators.length > 0) ? data.indicators[data.indicators.length - 1] : {};
        const pred = data.latest_prediction || {};

        const chatRes = await fetch('/api/v1/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: userPrompt || `Explain the chart and technical indicators for ${sym}.`,
                session_id: 'default_session',
                history: chatHistory,
                stock_analysis: {
                    symbol: sym,
                    current_price: latest.price,
                    ema_5: latest.ema_5,
                    rsi: latest.rsi,
                    macd: latest.macd,
                    vwap: latest.vwap,
                    p_long: pred.p_long,
                    p_short: pred.p_short,
                    action: pred.action
                }
            })
        });
        if (chatRes.ok) {
            const chatData = await chatRes.json();
            llmExplanation = chatData.answer || '';
            providerUsed = chatData.provider_used;
        }
    } catch (e) {
        console.warn('Failed to fetch LLM chart explanation:', e);
    }

    removeTyping(typingEl);
    const msgText = llmExplanation || `Loading technical charts for ${formatSymbolName(sym)}. You can switch between the Price / EMA, RSI, and MACD tabs below.`;
    appendAIMessage(msgText, null, providerUsed);
    updateSuggestionChips([
        'Explain 5 EMA vs VWAP',
        'Explain RSI momentum',
        'Explain MACD crossover',
        'Allocate capital to this stock'
    ]);
}


async function handleInvestIntent(intent, typingEl, userPrompt) {
    if (intent.symbol) {
        return await handleStockIntent(intent, typingEl, userPrompt);
    }
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

async function handleSectorIntent(intent, typingEl, userPrompt) {
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

    let llmExplanation = '';
    let providerUsed = null;
    try {
        const chatRes = await fetch('/api/v1/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: userPrompt || `Compare ${sector} sector stocks.`,
                session_id: 'default_session',
                history: chatHistory
            })
        });
        if (chatRes.ok) {
            const chatData = await chatRes.json();
            llmExplanation = chatData.answer || '';
            providerUsed = chatData.provider_used;
        }
    } catch (e) {
        console.warn('Failed to fetch sector LLM explanation:', e);
    }

    removeTyping(typingEl);

    if (!stocks.length) {
        appendAIMessage(llmExplanation || `I don't have a predefined stock list for that sector. Try asking about a specific stock like "Tell me about Reliance".`, null, providerUsed);
        return;
    }

    const defaultMsg = `The ${sector.toUpperCase()} sector on NSE includes: <strong>${stocks.map(formatSymbolName).join(', ')}</strong>. I can pull live data for any of these — just ask about one specifically, or tell me you want to invest and I'll scan the full sector.`;
    const responseText = llmExplanation ? `${llmExplanation}<br/><br/>${defaultMsg}` : defaultMsg;

    appendAIMessage(
        responseText,
        `<div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;">
            ${stocks.map(s => `<button class="chip" onclick="sendSuggestion('Tell me about ${formatSymbolName(s)}')">${formatSymbolName(s)}</button>`).join('')}
        </div>`,
        providerUsed
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

async function handleMarketIntent(typingEl, userPrompt) {
    removeTyping(typingEl);
    let llmExplanation = '';
    let providerUsed = null;
    try {
        const chatRes = await fetch('/api/v1/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: userPrompt || 'What is the market looking like today?',
                session_id: 'default_session',
                history: chatHistory
            })
        });
        if (chatRes.ok) {
            const chatData = await chatRes.json();
            llmExplanation = chatData.answer || '';
            providerUsed = chatData.provider_used;
        }
    } catch (e) {
        console.warn('Failed to fetch market overview LLM explanation:', e);
    }

    const hour = new Date().getHours();
    const marketOpen = hour >= 9 && hour < 16;
    const status = marketOpen ? 'NSE is currently within trading hours (9:15 AM – 3:30 PM IST)' : 'NSE markets are closed for regular trading sessions';
    const defaultMsg = `${status}. I can analyse live price data, technical indicators, and sentiment for top NSE liquid equities.`;

    appendAIMessage(
        llmExplanation ? `${llmExplanation}<br/><br/>${defaultMsg}` : defaultMsg,
        `<div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;">
            <button class="chip" onclick="sendSuggestion('Tell me about HDFC Bank')">HDFC Bank</button>
            <button class="chip" onclick="sendSuggestion('Tell me about Reliance')">Reliance</button>
            <button class="chip" onclick="sendSuggestion('Tell me about TCS')">TCS</button>
            <button class="chip" onclick="sendSuggestion('Which banking stocks look strong?')">Banking sector</button>
        </div>`,
        providerUsed
    );
    updateSuggestionChips([
        'Tell me about HDFC Bank',
        'Tell me about Reliance',
        'Compare IT sector stocks',
        'I want to invest ₹50,000'
    ]);
}

async function handleGeneralIntent(msg, typingEl) {
    let responseAnswer = '';
    let providerUsed = null;
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
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        responseAnswer = data.answer || 'I can help you with NSE stock analysis, market conditions, and intraday capital allocation. Try asking about a specific stock or sector.';
        providerUsed = data.provider_used;
    } catch (err) {
        responseAnswer = "I couldn't reach the AI service right now. Your market-data analysis is still separate from the chat service.";
    } finally {
        removeTyping(typingEl);
        appendAIMessage(responseAnswer, null, providerUsed);
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

        // Fetch LLM explanation for market scan result
        let llmSummary = '';
        try {
            const chatRes = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: `Explain the market scan results for investment of Rs ${capital}.`,
                    session_id: 'default_session',
                    stock_analysis: {
                        scan_type: 'market_scan',
                        investment_amount: capital,
                        qualified_count: currentRecommendations.length,
                        watchlist_top: (data.watchlist || []).slice(0, 3).map(w => ({
                            symbol: w.symbol,
                            direction: w.direction,
                            model_probability: w.model_probability,
                            distance_to_threshold: w.distance_to_threshold
                        }))
                    }
                })
            });
            if (chatRes.ok) {
                const chatData = await chatRes.json();
                llmSummary = chatData.answer || '';
            }
        } catch (e) {
            console.warn('Failed to fetch LLM scan summary:', e);
        }

        // AI message summary
        const topPick = currentRecommendations[0];
        if (topPick) {
            appendAIMessage(
                `Scan complete. I found <strong>${currentRecommendations.length} high-confidence trade setups</strong> across your ₹${capital.toLocaleString('en-IN')} allocation. The top pick is <strong>${formatSymbolName(topPick.symbol)}</strong> — ${topPick.action} signal at ${(topPick.confidence || 0).toFixed(2)}% Model probability, with ₹${topPick.allocated_capital?.toFixed(2)} allocated.` +
                (llmSummary ? `<br/><br/>${llmSummary}` : ''),
                null
            );
        } else {
            const topWatch = (data.watchlist || [])[0];
            const watchMsg = topWatch ? ` ${topWatch.symbol.replace('.NS','')} is the strongest monitored setup at ${(topWatch.model_probability*100).toFixed(2)}% (${topWatch.direction}), but it remains ${topWatch.distance_to_threshold.toFixed(2)} percentage points below the required threshold.` : '';
            appendAIMessage(
                `No trade currently meets the 80% model threshold across your ₹${capital.toLocaleString('en-IN')} capital input.<br/><br/>` +
                `• Allocated: <strong>₹0.00</strong><br/>` +
                `• Cash Reserve: <strong>₹${capital.toLocaleString('en-IN')}</strong> (100% full investment amount preserved)<br/><br/>` +
                (llmSummary || `No capital allocated because no candidate crossed the strict 80% model threshold.${watchMsg}`),
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

// ── Render Recommendations & Watchlist ──────────────────────────────────────
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

    // 1. Qualified Recommendation cards
    const container = document.getElementById('recommendations-container');
    container.innerHTML = '';

    const recs = data.recommendations || [];
    if (recs.length === 0) {
        container.innerHTML = `
            <div style="grid-column:1/-1;background:var(--surface);border:1px dashed var(--border);border-radius:14px;padding:24px;text-align:center;">
                <div style="font-size:0.95rem;font-weight:700;color:var(--text);margin-bottom:6px;">No trade currently meets the 80% model threshold.</div>
                <div style="font-size:0.8rem;color:var(--text-subtle);line-height:1.5;">No capital allocated because no candidate crossed the strict 80% model threshold. Your full capital (₹${data.investment_amount?.toLocaleString('en-IN')}) is preserved in cash reserve.</div>
            </div>
        `;
    } else {
        recs.forEach(rec => {
            const isBuy = rec.action === 'BUY';
            const card = document.createElement('div');
            card.className = `rec-card ${isBuy ? 'buy-card' : 'sell-card'}`;
            const probPct = rec.model_probability ? (rec.model_probability * 100).toFixed(2) : (rec.confidence || 0).toFixed(2);
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
                        <span>Model Probability</span>
                        <span style="font-weight:600;color:${isBuy?'var(--green)':'var(--red)'};">${probPct}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:${probPct}%;background:${isBuy?'var(--green)':'var(--red)'};"></div>
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
    }

    // 2. Watchlist cards (Top Setups to Monitor)
    const watchContainer = document.getElementById('watchlist-container');
    if (watchContainer) {
        watchContainer.innerHTML = '';
        const watchlist = data.watchlist || [];
        if (watchlist.length === 0) {
            document.getElementById('watchlist-section').style.display = 'none';
        } else {
            document.getElementById('watchlist-section').style.display = 'block';
            watchlist.forEach(item => {
                const card = document.createElement('div');
                card.className = 'rec-card';
                card.style.borderColor = 'var(--border)';
                const probPct = (item.model_probability * 100).toFixed(2);
                const gapPct = item.distance_to_threshold.toFixed(2);
                const threshPct = (item.model_threshold * 100).toFixed(2);

                card.innerHTML = `
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;position:relative;z-index:1;">
                        <div>
                            <div style="font-size:0.72rem;font-weight:600;color:var(--text-subtle);letter-spacing:0.04em;text-transform:uppercase;margin-bottom:4px;">${item.symbol}</div>
                            <div style="font-size:1.2rem;font-weight:700;letter-spacing:-0.02em;color:var(--text);">₹${item.current_price?.toFixed(2)}</div>
                        </div>
                        <div style="text-align:right;">
                            <span class="chip" style="font-size:0.7rem;padding:2px 8px;font-weight:700;color:var(--accent);border-color:var(--border);">${item.direction}</span>
                            <div style="font-size:0.65rem;font-weight:700;color:var(--text-subtle);margin-top:4px;letter-spacing:0.04em;text-transform:uppercase;">STATUS: NOT QUALIFIED</div>
                        </div>
                    </div>
                    <div style="margin-bottom:12px;position:relative;z-index:1;">
                        <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:var(--text-subtle);margin-bottom:4px;">
                            <span>Model Probability</span>
                            <span style="font-weight:600;color:var(--text);">${probPct}%</span>
                        </div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:${probPct}%;background:var(--accent);"></div>
                        </div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.75rem;margin-bottom:12px;position:relative;z-index:1;">
                        <div>
                            <div style="color:var(--text-subtle);font-size:0.65rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">Threshold</div>
                            <div style="font-weight:600;color:var(--text);">${threshPct}%</div>
                        </div>
                        <div>
                            <div style="color:var(--text-subtle);font-size:0.65rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">Gap</div>
                            <div style="font-weight:600;color:var(--accent);">${gapPct} percentage points</div>
                        </div>
                        <div>
                            <div style="color:var(--text-subtle);font-size:0.65rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">RSI</div>
                            <div style="font-weight:600;color:var(--text);">${item.rsi !== undefined && item.rsi !== null ? item.rsi.toFixed(1) : 'N/A'}</div>
                        </div>
                        <div>
                            <div style="color:var(--text-subtle);font-size:0.65rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">MACD</div>
                            <div style="font-weight:600;color:var(--text);">${item.macd !== undefined && item.macd !== null ? item.macd.toFixed(3) : 'N/A'}</div>
                        </div>
                        <div>
                            <div style="color:var(--text-subtle);font-size:0.65rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">Sentiment Score</div>
                            <div style="font-weight:600;color:var(--text);">${item.sentiment_score !== undefined && item.sentiment_score !== null ? item.sentiment_score.toFixed(3) : 'N/A'}</div>
                        </div>
                        <div>
                            <div style="color:var(--text-subtle);font-size:0.65rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">Market Sim</div>
                            <div style="font-weight:600;color:var(--text);">${item.market_similarity !== undefined && item.market_similarity !== null ? item.market_similarity.toFixed(4) : '0.0000'}</div>
                        </div>
                        <div>
                            <div style="color:var(--text-subtle);font-size:0.65rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">Stock Sim</div>
                            <div style="font-weight:600;color:var(--text);">${item.stock_similarity !== undefined && item.stock_similarity !== null ? item.stock_similarity.toFixed(4) : '0.0000'}</div>
                        </div>
                        <div>
                            <div style="color:var(--text-subtle);font-size:0.65rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">Allocation</div>
                            <div style="font-weight:600;color:var(--text-subtle);">₹0.00</div>
                        </div>
                    </div>
                    <div style="margin-top:10px;position:relative;z-index:1;">
                        <button class="btn-ghost" style="font-size:0.75rem;width:100%;text-align:center;" onclick="requestCharts('${item.symbol}')">View charts</button>
                    </div>
                `;
                watchContainer.appendChild(card);
            });
        }
    }


    // 3. Doughnut chart
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

function renderMarkdownToHTML(text) {
    if (!text) return '';
    let str = text;

    // 1. Expand squished markdown tables where newlines became ||
    str = str.replace(/\|\|/g, '\n|');

    // 2. Parse Markdown tables into clean HTML tables
    str = str.replace(/(?:(?:\|.+?\|(?:\r?\n|\r)?)+)/g, (match) => {
        const lines = match.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
        if (lines.length < 2) return match;

        let html = '<div style="overflow-x:auto;margin:12px 0;"><table class="data-table" style="width:100%;font-size:0.8rem;border-collapse:collapse;background:var(--bg-secondary);border-radius:8px;">';
        let isHeader = true;

        lines.forEach((line) => {
            if (/^\|?\s*[-:]+[-|\s:]*\|?$/.test(line)) return;
            const cells = line.split('|').map(c => c.trim()).filter((c, idx, arr) => (idx > 0 && idx < arr.length - 1) || (arr.length <= 2 && c !== ''));
            if (cells.length === 0) return;

            if (isHeader) {
                html += '<thead><tr style="border-bottom:1px solid var(--border);background:var(--surface);">';
                cells.forEach(c => { html += `<th style="padding:8px 12px;text-align:left;font-size:0.75rem;color:var(--text-subtle);text-transform:uppercase;">${c}</th>`; });
                html += '</tr></thead><tbody>';
                isHeader = false;
            } else {
                html += '<tr>';
                cells.forEach(c => { html += `<td style="padding:8px 12px;border-bottom:1px solid var(--border-soft);">${c}</td>`; });
                html += '</tr>';
            }
        });
        html += '</tbody></table></div>';
        return html;
    });

    // 3. Headings ### -> <h4>
    str = str.replace(/###\s+(.+?)(?=\n|$)/g, '<h4 style="font-size:0.9rem;font-weight:700;color:var(--text);margin-top:16px;margin-bottom:8px;">$1</h4>');
    str = str.replace(/##\s+(.+?)(?=\n|$)/g, '<h3 style="font-size:0.95rem;font-weight:700;color:var(--text);margin-top:18px;margin-bottom:8px;">$1</h3>');

    // 4. Split unformatted inline points " 1. **Header**" or " 2. **Header**" onto separate lines
    str = str.replace(/\s+(\d+\.\s+\*\*)/g, '\n$1');
    str = str.replace(/\s+([\*\-•]\s+\*\*)/g, '\n$1');

    // 5. Convert bullet points and numbered points into HTML <ul>/<ol> lists
    const lines = str.split(/\r?\n/);
    let inList = false;
    let listType = 'ul';
    const processed = [];

    lines.forEach(line => {
        const trimmed = line.trim();
        const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
        const bulletMatch = trimmed.match(/^[\*\-•]\s+(.*)/);

        if (numMatch) {
            if (!inList || listType !== 'ol') {
                if (inList) processed.push(`</${listType}>`);
                processed.push('<ol style="margin:10px 0;padding-left:22px;display:flex;flex-direction:column;gap:8px;">');
                inList = true;
                listType = 'ol';
            }
            processed.push(`<li style="line-height:1.6;color:var(--text);">${numMatch[2]}</li>`);
        } else if (bulletMatch) {
            if (!inList || listType !== 'ul') {
                if (inList) processed.push(`</${listType}>`);
                processed.push('<ul style="margin:10px 0;padding-left:22px;display:flex;flex-direction:column;gap:8px;">');
                inList = true;
                listType = 'ul';
            }
            processed.push(`<li style="line-height:1.6;color:var(--text);">${bulletMatch[1]}</li>`);
        } else {
            if (inList) {
                processed.push(`</${listType}>`);
                inList = false;
            }
            processed.push(line);
        }
    });
    if (inList) processed.push(`</${listType}>`);

    str = processed.join('\n');

    // 6. Convert Bold **text** -> <strong>text</strong>
    str = str.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // 7. Convert remaining newlines into <br/>
    str = str.replace(/\n\n+/g, '<br/><br/>').replace(/\n/g, '<br/>');

    // Clean up excessive <br/> next to block elements
    str = str.replace(/(?:<br\s*\/?>\s*)+(<(?:h[1-6]|table|div|ul|ol))/gi, '$1');
    str = str.replace(/(<\/(?:h[1-6]|table|div|ul|ol)>)(?:\s*<br\s*\/?>)+/gi, '$1');

    return str;
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

    const formattedContent = renderMarkdownToHTML(htmlText);

    div.innerHTML = `<div style="font-size:0.85rem;color:var(--text);line-height:1.65;">${formattedContent}</div>${extraHtml ? `<div style="margin-top:0;">${extraHtml}</div>` : ''}${badgeHtml}`;
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
