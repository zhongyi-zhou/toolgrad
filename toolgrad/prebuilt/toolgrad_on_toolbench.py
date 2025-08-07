from langgraph.graph import END, START, StateGraph
from langchain_community.callbacks.manager import get_openai_callback
from toolgrad.modules import graph_lib
import functools


# TODO:
# - llm name
# - a list of tool as the protocol
def create_graph_on_toolbench(
    sample_seed: int,
    num_apis: int = 50,
    num_iterations: int = 10,
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
      ),
  )
  graph.add_node("api_proposer", graph_lib.api_proposer)
  graph.add_node("api_executor", graph_lib.api_executor)
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
