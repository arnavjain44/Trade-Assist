import httpx
import logging
from typing import Optional, Dict, Any, List, Tuple
from app.config import settings

logger = logging.getLogger(__name__)

# Fallback financial knowledge base for educational queries when offline / no API key is provided
FINANCIAL_KNOWLEDGE_BASE = {
    "ema": (
        "The **5 EMA (Exponential Moving Average)** is a short-term trend indicator that calculates "
        "the average price of a stock over the last 5 periods, giving more weight to recent prices. "
        "In intraday trading, when the price crosses **above** the 5 EMA, it indicates immediate bullish momentum. "
        "When the price drops **below** the 5 EMA, it signals weakening short-term strength."
    ),
    "rsi": (
        "The **RSI (Relative Strength Index)** measures the speed and change of price movements on a scale of 0 to 100. "
        "Readings **above 70** suggest a stock is overbought (potential price pullback), while readings **below 30** "
        "indicate oversold conditions (potential price bounce). Values between 40 and 60 represent neutral momentum."
    ),
    "macd": (
        "The **MACD (Moving Average Convergence Divergence)** tracks trend direction and momentum. "
        "It consists of the MACD line (difference between 12-period and 26-period EMAs) and a Signal line (9-period EMA of MACD). "
        "A **bullish crossover** (MACD crossing above Signal) indicates a buying opportunity, while a **bearish crossover** "
        "signals selling pressure."
    ),
    "vwap": (
        "**VWAP (Volume Weighted Average Price)** provides the average price a stock has traded at throughout the day, "
        "based on both volume and price. It acts as an institutional benchmark. Trading **above VWAP** indicates intraday "
        "buyers are in control; trading **below VWAP** suggests sellers are dominant."
    ),
    "bollinger": (
        "**Bollinger Bands** consist of a 20-day simple moving average with upper and lower volatility bands (2 standard deviations). "
        "When bands contract, volatility is low and a sharp breakout often follows. Touching the upper band suggests overbought "
        "conditions, while the lower band indicates oversold levels."
    ),
    "obv": (
        "**OBV (On-Balance Volume)** measures cumulative volume flow to predict price movements. Rising OBV shows smart money "
        "accumulating shares on up days, confirming bullish momentum. Falling OBV signals distribution."
    ),
    "stop loss": (
        "A **Stop Loss** is a predetermined exit price designed to cap maximum loss on a trade if the market moves against you. "
        "In our intraday system, stop losses are calculated dynamically below recent support levels."
    ),
    "target": (
        "A **Target Price** is the projected exit price where profits are taken, based on risk-reward ratios and resistance levels."
    ),
    "risk": (
        "Risk management is strictly enforced in this system: No single stock receives more than **40% of total capital**, "
        "and all positions have a **mandatory same-day exit rule** before market close."
    ),
    "intraday": (
        "**Intraday Trading** involves buying and selling stock positions within the same trading day before market close (3:30 PM IST on NSE). "
        "No positions are carried overnight, eliminating overnight gap risk."
    ),
}


def detect_provider_override(text: str) -> Tuple[Optional[str], List[str]]:
    """
    Detects if a user prompt explicitly requests a specific LLM provider (e.g. 'use gemini', 'via groq', '@openrouter').
    Returns Tuple[requested_provider_name_or_None, list_of_providers_in_priority_order].

    Default priority order: ['gemini', 'groq', 'openrouter'].
    If an override is detected, that provider is tried FIRST while preserving remaining providers as fallbacks.
    """
    default_order = ["gemini", "groq", "openrouter"]
    if not text:
        return None, default_order

    lower = text.lower()

    # Provider keyword match maps
    provider_triggers = {
        "gemini": ["gemini", "@gemini", "google gemini"],
        "groq": ["groq", "@groq", "llama"],
        "openrouter": ["openrouter", "@openrouter"]
    }

    detected = None
    for p_name, keywords in provider_triggers.items():
        if any(kw in lower for kw in keywords):
            detected = p_name
            break

    if detected and detected in default_order:
        # Reorder priority list so requested provider is tried first, followed by fallbacks
        priority_order = [detected] + [p for p in default_order if p != detected]
        return detected, priority_order

    return None, default_order


class LLMClient:
    """Handles agentic LLM queries using Gemini / Groq / OpenRouter API when available,
    with an intelligent financial knowledge base fallback when offline or no key is provided.
    """

    async def generate_response(
        self,
        prompt: str,
        system_context: str = "",
        provider_choice: str = "auto",
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, str]:
        """Attempts live LLM generation iterating through key pools with multi-key failover.
        Returns Tuple[response_text, provider_actually_used].
        """
        # 1. Detect provider override from user prompt text
        override_provider, priority_order = detect_provider_override(prompt)

        # 2. If no prompt override, check provider_choice argument
        if not override_provider and provider_choice in ["gemini", "groq", "openrouter"]:
            priority_order = [provider_choice] + [p for p in ["gemini", "groq", "openrouter"] if p != provider_choice]

        # Iteratively attempt APIs in priority order
        for provider in priority_order:
            if provider == "gemini":
                gemini_keys = settings.get_gemini_key_pool()
                for key in gemini_keys:
                    res = await self._call_gemini_api(prompt, system_context, key, history)
                    if res:
                        return res, "gemini"
                    logger.warning("Gemini key failed or rate-limited; attempting next key in pool...")

            elif provider == "groq":
                groq_keys = settings.get_groq_key_pool()
                for key in groq_keys:
                    res = await self._call_groq_api(prompt, system_context, key, history)
                    if res:
                        return res, "groq"
                    logger.warning("Groq key failed or rate-limited; attempting next key in pool...")

            elif provider == "openrouter":
                openrouter_keys = settings.get_openrouter_key_pool()
                for key in openrouter_keys:
                    res = await self._call_openrouter_api(prompt, system_context, key, history)
                    if res:
                        return res, "openrouter"
                    logger.warning("OpenRouter key failed or rate-limited; attempting next key in pool...")

        # 3. Fallback: Intelligent Agentic Knowledge Router
        fallback_res = self._knowledge_fallback(prompt, system_context)
        return fallback_res, "offline_knowledge_base"



    async def _call_gemini_api(self, prompt: str, system_context: str, api_key: str, history: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        full_text = f"{system_context}\n\nUser Question: {prompt}" if system_context else prompt
        payload = {
            "contents": [{"parts": [{"text": full_text}]}]
        }
        logger.info("[LLM CALL] Invoking Provider: GOOGLE GEMINI (gemini-1.5-flash)...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                logger.info("[LLM RESPONSE] Provider: GOOGLE GEMINI | Status Code: %d", resp.status_code)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                else:
                    logger.warning("[LLM ERROR] Gemini API returned HTTP %d: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("[LLM ERROR] Gemini API exception: %s", e)
        return None

    async def _call_groq_api(self, prompt: str, system_context: str, api_key: str, history: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        if history:
            for turn in history[-6:]:
                role = "assistant" if turn.get("role") == "agent" or turn.get("agent") else "user"
                content = turn.get("content") or turn.get("agent") or turn.get("user")
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.5,
        }
        logger.info("[LLM CALL] Invoking Provider: GROQ (llama-3.3-70b-versatile)...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                logger.info("[LLM RESPONSE] Provider: GROQ | Status Code: %d", resp.status_code)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                else:
                    logger.warning("[LLM ERROR] Groq API returned HTTP %d: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("[LLM ERROR] Groq API exception: %s", e)
        return None

    async def _call_openrouter_api(self, prompt: str, system_context: str, api_key: str, history: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        if history:
            for turn in history[-6:]:
                role = "assistant" if turn.get("role") == "agent" or turn.get("agent") else "user"
                content = turn.get("content") or turn.get("agent") or turn.get("user")
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "google/gemini-2.0-flash-exp:free",
            "messages": messages,
        }
        logger.info("[LLM CALL] Invoking Provider: OPENROUTER (google/gemini-2.0-flash-exp:free)...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                logger.info("[LLM RESPONSE] Provider: OPENROUTER | Status Code: %d", resp.status_code)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                else:
                    logger.warning("[LLM ERROR] OpenRouter API returned HTTP %d: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("[LLM ERROR] OpenRouter API exception: %s", e)
        return None

    def _knowledge_fallback(self, prompt: str, system_context: str) -> str:
        """Determines intent and returns tailored financial explanations when offline."""
        logger.info("[LLM FALLBACK TRIGGERED] Provider API keys unconfigured/failed — Using Offline Knowledge Base Router for prompt: '%s'", prompt[:60])
        lower = prompt.lower().strip()

        # Chart explanation requests
        if "chart" in lower or "indicator" in lower or "trend" in lower:
            return (
                "Our technical chart analysis plots 6 intraday indicators:\n\n"
                "• **5 EMA vs Price**: Detects immediate short-term trend direction. Price above 5 EMA signals bullish momentum.\n"
                "• **VWAP (Volume Weighted Average Price)**: Institutional benchmark. Trading above VWAP shows strong buying conviction.\n"
                "• **RSI (Relative Strength Index)**: Momentum indicator. Above 70 is overbought (pullback risk); below 30 is oversold (bounce potential).\n"
                "• **MACD Crossover**: Difference between 12 & 26 EMAs. MACD crossing above Signal line indicates a bullish entry trigger.\n\n"
                "To view interactive charts for any stock, click **'View technical charts'** on any stock card or select it in the Technical Charts panel."
            )

        # Check knowledge base matches
        for key, explanation in FINANCIAL_KNOWLEDGE_BASE.items():
            if key in lower:
                return explanation

        # Context-aware follow-ups (if a recent scan occurred)
        if system_context:
            if "why" in lower or "reason" in lower or "picked" in lower:
                return f"Based on your session data:\n{system_context}"

        return (
            "I am your AI Intraday Trading Assistant. You can ask me to analyze any NSE stock (e.g. Reliance, HDFC Bank, TCS, Zomato, Tata Steel), "
            "explain technical indicators like 5 EMA, RSI, MACD, or VWAP, or specify an investment amount to generate a full intraday allocation plan."
        )



llm_client = LLMClient()
