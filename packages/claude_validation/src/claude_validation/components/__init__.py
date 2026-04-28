import os, sys, anthropic
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
                    return {
                        "action":    block.input["action"],
                        "reasoning": block.input["reasoning"]
                    }

            return {"action": dqn_suggestion, "reasoning": "Claude fallback to DQN"}
        except Exception as e:
            raise CustomException(e, sys)
        
    def _build_prompt(self, metrics: dict, dqn_suggestion: str, dqn_confidence: float) -> str:
        return f"""
        You are an AI system monitor for a medical image segmentation service.
        Given these metrics, decide the best action.

        Current metrics:
        - CPU:         {metrics['cpu']}%
        - RAM:         {metrics['ram']}%
        - Latency:     {metrics['latency']}s
        - Drift score: {metrics['drift_score']} (threshold: {self.config.params.confidence_threshold})

        DQN suggestion: {dqn_suggestion} (confidence: {dqn_confidence:.2f})
        Low confidence means DQN is uncertain — use your judgment.

        Choose the most appropriate action and explain why.
        """.strip()