import argparse
import concurrent.futures
import logging
import os
import threading

from langchain_community.callbacks import get_openai_callback
import pandas as pd
from scipy import stats
import tqdm

from src import eval_lib
from src.utils import common_utils
from src.utils import metrics_utils


class Evaluator:

  def __init__(self, dataset_dir: str, judge_name: str, unbiased_writer: str):
    self.dataset_dir = dataset_dir
    self.GT_SFT_DIR = os.path.join(dataset_dir, "sft_data")
    self.GT_RAW_DIR = os.path.join(dataset_dir, "data")
    self.judge = eval_lib.create_llm_judge(judge_name)
    self.recall = metrics_utils.AverageStrListMeter()
    self.gt_tool_success = metrics_utils.AverageMeter()
    self.qor_dict = {
        f"response-unbiased-{unbiased_writer}": metrics_utils.AverageMeter(),
    }
    self.test_samples = common_utils.read_list_from_txt(
        f'{dataset_dir}/sft_data/test.txt')
    self.test_samples.sort()
    self.lock = threading.Lock()

  def compute_tool_recall(self, pred_data, gt_metadata):
    pred_tool_names = {
        api_use_tuple[0]['tool']
        for api_use_tuple in pred_data['intermediate_steps']
    }
    gt_tool_names = set(gt_metadata['pos_tools'].keys())
    with self.lock:
      acc_dict = self.recall.update(gt=gt_tool_names, pred=pred_tool_names)
    return acc_dict

  def compute_recalled_tool_success(
      self,
      pred_data,
      gt_metadata,
  ):
    gt_tool_names = set(gt_metadata['pos_tools'].keys())

    gt_tool_call_success = {k: 0 for k in gt_tool_names}
    for api_use_tuple in pred_data['intermediate_steps']:
      if api_use_tuple[0]['tool'] in gt_tool_call_success.keys():
        toolname = api_use_tuple[0]['tool']
        if gt_tool_call_success[toolname] == 1:
          continue
        if api_use_tuple[1]['error'] == "":
          gt_tool_call_success[toolname] = 1
    success_rate = sum(
        gt_tool_call_success.values()) / len(gt_tool_call_success)
    with self.lock:
      self.gt_tool_success.update(success_rate)
    return success_rate

  def compute_response_score(
      self,
      pred_data,
      gt_data,
      pred_response_key: str = "response-base",
  ) -> int:
    judge_result = self.judge.invoke({
        "query": gt_data['query'],
        "tool_use_trace": pred_data['intermediate_steps'],
        "pred": pred_data[pred_response_key],
        "gt": gt_data['response'],
    })
    assert isinstance(judge_result, eval_lib.EvaluationResult)
    score = judge_result.final_score
    score_clip = eval_lib.clip_value(score, 0, 100)
    with self.lock:
      self.qor_dict[pred_response_key].update(score_clip)
    return score_clip

  def run_one_atN(
      self,
      model_name: str,
      bench_str: str,
      num_workers: int = 8,
      overwrite: bool = False,
      # ) -> tuple[float, float, float]:
  ) -> tuple[float, float]:
    pred_dir = f"{self.dataset_dir}/prediction/{model_name}/"
    self.recall.reset()
    for m in self.qor_dict.values():
      m.reset()

    # Create a lock to ensure thread-safe updates to shared objects
    lock = threading.Lock()

    def process_sample(sample_id, overwrite=False):
      pred_data = common_utils.read_json(
          f"{pred_dir}/{bench_str}/{sample_id}.json")
      gt_metadata = common_utils.read_json(
          f"{self.GT_SFT_DIR}/{bench_str}/metadata/{sample_id}.json")
      gt_data = common_utils.read_json(f"{self.GT_RAW_DIR}/{sample_id}.json")

      acc = None
      success_rate = None
      qor = None
      if "eval" in pred_data and not overwrite:
        # Skip samples that have already been evaluated
        logging.info(f"Skipping {sample_id} as it has already been evaluated.")
        if "tool_accuracy" in pred_data["eval"]:
          precision = pred_data["eval"]["tool_accuracy"]["precision"]
          recall = pred_data["eval"]["tool_accuracy"]["recall"]
          acc = {
              "precision": precision,
              "recall": recall,
          }
          with lock:
            self.recall.update(precision=precision, recall=recall)
        if "success_rate" in pred_data["eval"]:
          success_rate = pred_data["eval"]["success_rate"]
          with lock:
            self.gt_tool_success.update(success_rate)
        if "qor" in pred_data["eval"]:
          qor = pred_data["eval"]["qor"]
          with lock:
            self.qor_dict[f"response-unbiased-{args.unbiased_writer}"].update(
                qor)

      if acc is not None and success_rate is not None and qor is not None:
        # If all metrics are already computed, skip processing
        logging.info(
            f"Skipping {sample_id} as all metrics are already computed.")
        return
      if pred_data == {} or "error" in pred_data:
        # Use lock to update shared meters safely
        with lock:
          self.recall.update(gt=['1'], pred=['2'])
          for m in self.qor_dict.values():
            m.update(0)
        logging.info(f"Skipping {sample_id} due to error in prediction.")
        return
      if acc is None:
        acc = self.compute_tool_recall(pred_data, gt_metadata)
        with lock:
          self.recall.update(**acc)
      if success_rate is None:
        success_rate = self.compute_recalled_tool_success(
            pred_data, gt_metadata)
        with lock:
          self.gt_tool_success.update(success_rate)
      if qor is None:

        qor = self.compute_response_score(
            pred_data,
            gt_data,
            pred_response_key=f"response-unbiased-{args.unbiased_writer}",
        )
        with lock:
          self.qor_dict[f"response-unbiased-{args.unbiased_writer}"].update(qor)
      pred_with_eval = {
          **pred_data, "eval": {
              "tool_accuracy": acc,
              "success_rate": success_rate,
              "qor": qor,
          }
      }
      common_utils.save_json(
          pred_with_eval,
          f"{pred_dir}/{bench_str}/{sample_id}.json",
      )

    if num_workers == 1:
      logging.info("Single-threaded processing")
      # Single-threaded processing
      for sample_id in tqdm.tqdm(self.test_samples):
        process_sample(sample_id, overwrite=overwrite)
    else:
      # Multi-threaded processing
      logging.info("Multi-threaded processing")
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_sample, sample_id, overwrite=overwrite)
            for sample_id in self.test_samples
        ]
        for _ in tqdm.tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures)):
          pass
    return (
        self.recall.recall,
        # self.qor_dict["response-base"].avg,
        self.gt_tool_success.avg,
        self.qor_dict[f"response-unbiased-{args.unbiased_writer}"].avg,
    )


def paired_t_test(eval1: Evaluator, eval2: Evaluator):
  results = {}

  # Recall
  recall1 = [d["recall"] for d in eval1.recall.vals]
  recall2 = [d["recall"] for d in eval2.recall.vals]
  t_stat, p_value = stats.ttest_rel(recall1, recall2)
  print(f"recall: t-statistic={t_stat}, p-value={p_value}")
  results["recall"] = {
      "t_stat": t_stat,
      "p_value": p_value,
  }

  # Success Rate
  success_rate1 = eval1.gt_tool_success.vals
  success_rate2 = eval2.gt_tool_success.vals
  t_stat, p_value = stats.ttest_rel(success_rate1, success_rate2)
  print(f"success_rate: t-statistic={t_stat}, p-value={p_value}")
  results["success_rate"] = {
      "t_stat": t_stat,
      "p_value": p_value,
  }

  # Perform paired t-test on the QoR scores
  qor1 = eval1.qor_dict[f"response-unbiased-{args.unbiased_writer}"].vals
  qor2 = eval2.qor_dict[f"response-unbiased-{args.unbiased_writer}"].vals
  t_stat, p_value = stats.ttest_rel(qor1, qor2)
  print(f"QoR: t-statistic={t_stat}, p-value={p_value}")
  results["qor"] = {
      "t_stat": t_stat,
      "p_value": p_value,
  }

  return results


def create_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="")
  parser.add_argument('--dataset_dir',
                      type=str,
                      default=os.path.expanduser('~/dataset/ToolGrad-5k/'))
  parser.add_argument('--judge', type=str, default="gpt-4.1")
  parser.add_argument('--pred_model', type=str, default="gemini-2.0-flash")
  parser.add_argument('--compare_model',
                      type=str,
                      default=None,
                      help="Compare with this model for stats analysis")
  parser.add_argument('--sampling_method',
                      choices=['embedding'],
                      default='embedding')
  parser.add_argument('--num_process', type=int, default=1)
  parser.add_argument('--unbiased_writer', default="gpt-4.1-mini")
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = create_args()
  evaluator = Evaluator(dataset_dir=args.dataset_dir,
                        judge_name=args.judge,
                        unbiased_writer=args.unbiased_writer)
  recall, success_rate, qor = evaluator.run_one_atN(
      args.pred_model,
      f"method={args.sampling_method}",
      num_workers=args.num_process,
      overwrite=args.overwrite,
  )
  results = {}
  df = pd.DataFrame({
      "Model": [args.pred_model],
      "Recall": [recall],
      "Success Rate": [success_rate],
      "QoR": [qor],
  })
  df = df.set_index("Model")  # Optionally, set model names as index
  logging.info(df)
  print(df)

  if args.compare_model is not None:
    with get_openai_callback() as cb:
      evaluator_2 = Evaluator(dataset_dir=args.dataset_dir,
                              judge_name=args.judge,
                              unbiased_writer=args.unbiased_writer)
      recall_2, success_rate_2, qor_2 = evaluator_2.run_one_atN(
          args.compare_model,
          f"method={args.sampling_method}",
          num_workers=args.num_process,
          overwrite=args.overwrite,
      )
      df = pd.DataFrame({
          "Model": [args.compare_model],
          "Recall": [recall_2],
          "Success Rate": [success_rate_2],
          "QoR": [qor_2],
      })
      print(df)
    results = paired_t_test(evaluator, evaluator_2)
