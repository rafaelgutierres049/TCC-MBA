"""
Grafo LangGraph do AgenticML.
Spec: seções 7.1, 7.2 e 7.3

Fluxo (caminho feliz):
  Agent1 → Agent2 → Agent3 → Agent4 → Agent5 → Agent6 → END

Curto-circuito (erro crítico):
  AgentN → [status == "error"] → Agent6 (modo erro) → END
"""

from langgraph.graph import END, StateGraph

from agents.agent1_ingestion import run_agent1
from agents.agent2_quality import run_agent2
from agents.agent3_features import run_agent3
from agents.agent4_interpretation import run_agent4
from agents.agent5_training import run_agent5
from agents.agent6_explanation import run_agent6
from core.state import AgenticMLState


def _route(state: AgenticMLState) -> str:
    """
    Roteador condicional aplicado após cada agente (exceto o último).
    Se status == "error", encaminha para Agent6 em modo erro (spec 7.2).
    Caso contrário, continua o fluxo normal.
    """
    return "agent6" if state["status"] == "error" else "continue"


def build_graph():
    """
    Constrói e compila o grafo LangGraph do AgenticML.
    Retorna um CompiledGraph pronto para .invoke().
    """
    graph = StateGraph(AgenticMLState)

    # Registrar os nós
    graph.add_node("agent1", run_agent1)
    graph.add_node("agent2", run_agent2)
    graph.add_node("agent3", run_agent3)
    graph.add_node("agent4", run_agent4)
    graph.add_node("agent5", run_agent5)
    graph.add_node("agent6", run_agent6)

    # Ponto de entrada
    graph.set_entry_point("agent1")

    # Arestas condicionais: erro → agent6 (curto-circuito), sucesso → próximo
    graph.add_conditional_edges(
        "agent1", _route, {"continue": "agent2", "agent6": "agent6"}
    )
    graph.add_conditional_edges(
        "agent2", _route, {"continue": "agent3", "agent6": "agent6"}
    )
    graph.add_conditional_edges(
        "agent3", _route, {"continue": "agent4", "agent6": "agent6"}
    )
    graph.add_conditional_edges(
        "agent4", _route, {"continue": "agent5", "agent6": "agent6"}
    )
    # Agent5 sempre vai para Agent6 
    graph.add_edge("agent5", "agent6")
    graph.add_edge("agent6", END)

    return graph.compile()
