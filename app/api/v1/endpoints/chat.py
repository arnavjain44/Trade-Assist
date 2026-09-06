from fastapi import APIRouter, HTTPException
from app.schemas.requests import ChatRequest
from app.schemas.responses import ChatResponse
from app.agent.loop import agent_loop

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Follow-up Q&A chat endpoint with the agent. Keeps conversation context
    and reasoning memory for both picked and rejected stocks.
    """
    try:
        answer, provider_used, context_summary = await agent_loop.generate_chat_response(
            user_message=request.message,
            session_id=request.session_id or "default_session",
            provider_choice=request.provider or "auto",
            history=request.history
        )

        return ChatResponse(
            answer=answer,
            session_id=request.session_id or "default_session",
            provider_used=provider_used,
            reasoning_context=context_summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent chat failure: {str(e)}")
