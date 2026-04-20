from core.logging import logger
from ai_manager.components.monitor import Monitor
from ai_manager.components.analyzer import Analyzer
from ai_manager.components.planner import Planner
from ai_manager.components.executer import Executer

class ComponentFactory:
    @staticmethod
    def create(component_name: str, **kwargs):
        components = {
            "monitor":       Monitor,
            "lstm_analyzer": Analyzer,
            "dqn_planner":   Planner,
            "executer":      Executer,
        }
        
        if component_name not in components:
            raise ValueError(f"Unknown component: {component_name}")
        
        return components[component_name](**kwargs)