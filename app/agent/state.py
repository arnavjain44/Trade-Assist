from typing import Dict, Any, List, Optional


class AgentStateMemory:
    """Stores agent reasoning for all considered stocks (picked & rejected) across chat sessions."""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def set_active_stock(self, session_id: str, symbol: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"picked_stocks": [], "rejected_stocks": [], "history": [], "active_stock": None}
        self.sessions[session_id]["active_stock"] = symbol

    def get_active_stock(self, session_id: str) -> Optional[str]:
        return self.sessions.get(session_id, {}).get("active_stock")

    def save_session_run(
        self,
        session_id: str,
        picked_stocks: List[Dict[str, Any]],
        rejected_stocks: List[Dict[str, Any]],
        total_capital: float
    ):
        active = picked_stocks[0]["symbol"] if picked_stocks else None
        if session_id not in self.sessions:
            self.sessions[session_id] = {"history": []}
        self.sessions[session_id].update({
            "picked_stocks": picked_stocks,
            "rejected_stocks": rejected_stocks,
            "total_capital": total_capital,
            "active_stock": active
        })

    def add_chat_history(self, session_id: str, user_msg: str, agent_response: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"picked_stocks": [], "rejected_stocks": [], "history": [], "active_stock": None}
        self.sessions[session_id]["history"].append({"user": user_msg, "agent": agent_response})

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {"picked_stocks": [], "rejected_stocks": [], "history": [], "active_stock": None})


agent_memory = AgentStateMemory()

