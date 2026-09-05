from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Intraday Trading Web App API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Full NIFTY 50 Equities Universe
    DEFAULT_NSE_TICKERS: List[str] = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "TATAMOTORS.NS", "BHARTIARTL.NS", "SBIN.NS", "AXISBANK.NS", "ITC.NS",
        "WIPRO.NS", "BAJFINANCE.NS", "MARUTI.NS", "LT.NS", "HCLTECH.NS",
        "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "KOTAKBANK.NS",
        "TATASTEEL.NS", "INDUSINDBK.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
        "ONGC.NS", "HDFCLIFE.NS", "SBILIFE.NS", "BAJAJ-AUTO.NS", "M&M.NS",
        "HEROMOTOCO.NS", "EICHERMOT.NS", "BPCL.NS", "IOC.NS", "DIVISLAB.NS",
        "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "NESTLEIND.NS",
        "HINDUNILVR.NS", "GRASIM.NS", "JSWSTEEL.NS", "HINDALCO.NS", "ADANIENT.NS",
        "ADANIPORTS.NS", "BEL.NS", "HAL.NS", "TRENT.NS", "ZOMATO.NS"
    ]

    
    # Free LLM API Keys & Provider Pool (single key or comma-separated list for rotation/fallback)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_API_KEYS: Optional[str] = None

    GROQ_API_KEY: Optional[str] = None
    GROQ_API_KEYS: Optional[str] = None

    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_API_KEYS: Optional[str] = None

    DEFAULT_PRIMARY_PROVIDER: str = "gemini"
    SUPPORTED_PROVIDERS: List[str] = ["auto", "gemini", "groq", "openrouter"]

    def get_gemini_key_pool(self) -> List[str]:
        keys = []
        if self.GEMINI_API_KEYS:
            keys.extend([k.strip() for k in self.GEMINI_API_KEYS.split(",") if k.strip()])
        if self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip() not in keys:
            keys.append(self.GEMINI_API_KEY.strip())
        return keys

    def get_groq_key_pool(self) -> List[str]:
        keys = []
        if self.GROQ_API_KEYS:
            keys.extend([k.strip() for k in self.GROQ_API_KEYS.split(",") if k.strip()])
        if self.GROQ_API_KEY and self.GROQ_API_KEY.strip() not in keys:
            keys.append(self.GROQ_API_KEY.strip())
        return keys

    def get_openrouter_key_pool(self) -> List[str]:
        keys = []
        if self.OPENROUTER_API_KEYS:
            keys.extend([k.strip() for k in self.OPENROUTER_API_KEYS.split(",") if k.strip()])
        if self.OPENROUTER_API_KEY and self.OPENROUTER_API_KEY.strip() not in keys:
            keys.append(self.OPENROUTER_API_KEY.strip())
        return keys

    
    # Constraint Enforcement Rules
    MAX_POSITION_ALLOCATION_PCT: float = 0.40  # 40% max per stock
    FORCE_SAME_DAY_EXIT: bool = True
    
    # Neo4j Settings (leave NEO4J_URI empty to disable — set in .env if Neo4j is running)
    NEO4J_URI: str = ""
    NEO4J_USER: str = ""
    NEO4J_PASSWORD: str = ""
    
    # ChromaDB Settings
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
