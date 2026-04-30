import os, sys, json, time, anthropic
from pydantic import BaseModel, Field
from enum import Enum

class ActionType(str, Enum):
    do_nothing           = "do_nothing"
    trigger_retraining   = "trigger_retraining"
    switch_to_lighter    = "switch_to_lighter_model"
    restart_service      = "restart_service"
    scale_out            = "scale_out_service"
    scale_in             = "scale_in_service"
    swap_model           = "swap_model_version"
    
class ExecuteAction(BaseModel):
    action:     ActionType  = Field(description="Action to execute")
    reasoning:  str         = Field(description="Why this action was chosen")


from core.logging import logger
from core.exception import CustomException
from claude_validation import ClaudeValidationConfig

class ClaudeValidation:
    def __init__(self, config: ClaudeValidationConfig):
        self.config  = config
        self.client  = self._get_client()
        self.tools   = self._get_tools()
        self.history = self._get_history()
    
    def _get_client(self):
        from dotenv import load_dotenv
        load_dotenv()
        
        try:
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            logger.info(f"Successfully loaded Anthropic client")
            return client
        except Exception as e:
            raise CustomException(e, sys)
        
    def _get_tools(self):
        return [anthropic.types.ToolParam(
            name        = "execute_action",
            description = "Execute a system action when metrics indicate a problem",
            input_schema= ExecuteAction.model_json_schema()
        )]
    
    def _get_history(self) -> list:
        history_path = self.config.history_path
        
        history_path.parent.mkdir(parents=True, exist_ok=True)
        if history_path.exists():
            return json.loads(history_path.read_text())
        return []
    
    def claude_validate(self, metrics: dict, dqn_suggestion: str, dqn_confidence: float) -> dict:
        try:
            response  = self.client.messages.create(
                model = self.config.params.model,
                max_tokens = self.config.params.max_tokens,
                tools = self.tools,
                messages = [{
                    "role": "user",
                    "content": self._build_prompt(metrics, dqn_suggestion, dqn_confidence)
                }]
            )       
        
            for block in response.content:
                if block.type == "tool_use":
                    result = block.input
                    
                    self.history.append({
                        "timestamp":  time.time(),
                        "metrics":    {k: v[0] for k, v in metrics.items()},
                        "dqn":        dqn_suggestion,
                        "claude":     result["action"],
                        "reasoning":  result["reasoning"][:150]
                    })
                    self._save_history()
                    
                    return {
                        "action":    result["action"],
                        "reasoning": result["reasoning"]
                    }

            return {"action": dqn_suggestion, "reasoning": "Claude fallback to DQN"}
        except Exception as e:
            raise CustomException(e, sys)
        
    def _build_prompt(self, metrics: dict, dqn_suggestion: str, dqn_confidence: float) -> str:
        history_str = ""
        if self.history:
            recent = self.history[-5:]
            history_str = f"\nPrevious decisions:\n{json.dumps(recent, indent=2)}\n"
        
        return f"""
            You are an AI system monitor for a medical image segmentation service.
            {history_str}
            Current metrics:
            - CPU:         {metrics.get('current_cpu', [0.0])[0]}%
            - RAM:         {metrics.get('current_ram', [0.0])[0]} MB
            - Latency:     {metrics.get('current_latency', [0.0])[0]}s
            - Drift score: {metrics.get('current_drift', [0.0])[0]} (threshold: {self.config.params.confidence_threshold})

            DQN suggestion: {dqn_suggestion} (confidence: {dqn_confidence:.2f})
            Low confidence means DQN is uncertain — use your judgment.

            Choose the most appropriate action and explain why.
        """.strip()
        
    def _save_history(self):
        self.config.history_path.write_text(
            json.dumps(self.history[-20:], indent=2)
        )