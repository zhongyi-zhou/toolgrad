import functools
import os

from langchain_community.callbacks.manager import get_openai_callback
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

from toolgrad.modules import graph_lib
from toolgrad.utils import mcp


def create_graph_on_mcp(
    sample_seed: int,
    num_apis: int = 5,
    num_iterations: int = 5,
    mcp_dict: dict = None,
) -> StateGraph:
  graph = StateGraph(graph_lib.ToolGradState)

  # Add nodes
  graph.add_node(
      "sample_apis",
      lambda state: graph_lib.sample_apis_or_end(
          state,
          api_sample_seed=sample_seed,
          max_iter=num_iterations,
          num_apis=num_apis,
          database="mcp",
          mcp_dict=mcp_dict,
      ),
  )

  graph.add_node(
      "api_proposer",
      lambda state: graph_lib.api_proposer(state, is_toolbench=False))
  graph.add_node(
      "api_executor",
      lambda state: graph_lib.api_executor(state, is_toolbench=False))
  graph.add_node("api_selector", graph_lib.api_selector)
  graph.add_node("inverse_predictor", graph_lib.inverse_predictor)

  # Add edges
  graph.add_edge(START, "sample_apis")
  graph.add_edge("sample_apis", "api_proposer")
  graph.add_conditional_edges(
      "api_proposer",
      functools.partial(
          graph_lib.should_early_next_iter_or_end,
          max_iter=num_iterations,
      ),
      {
          graph_lib.SpecialStates.NEXT_STEP: "api_executor",
          graph_lib.SpecialStates.NEXT_ITER: "sample_apis",
          graph_lib.SpecialStates.END: END,
      },
  )
  graph.add_conditional_edges(
      "api_executor",
      functools.partial(
          graph_lib.should_early_next_iter_or_end,
          max_iter=num_iterations,
      ),
      {
          graph_lib.SpecialStates.NEXT_STEP: "api_selector",
          graph_lib.SpecialStates.NEXT_ITER: "sample_apis",
          graph_lib.SpecialStates.END: END,
      },
  )
  graph.add_conditional_edges(
      "api_selector",
      functools.partial(
          graph_lib.should_early_next_iter_or_end,
          max_iter=num_iterations,
      ),
      {
          graph_lib.SpecialStates.NEXT_STEP: "inverse_predictor",
          graph_lib.SpecialStates.NEXT_ITER: "sample_apis",
          graph_lib.SpecialStates.END: END,
      },
  )

  graph.add_conditional_edges(
      "inverse_predictor",
      functools.partial(graph_lib.should_end, max_iter=num_iterations),
      {
          graph_lib.SpecialStates.NEXT_ITER: "sample_apis",
          graph_lib.SpecialStates.END: END,
      },
  )

  app = graph.compile()
  return app
