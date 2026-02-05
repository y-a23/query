# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Follow this strict protocol to answer the question:"
    "1. Internal Reasoning: Every time you receive new information "
    "(either the initial question or search results), "
    "you must first conduct reasoning. Enclose this reasoning process between <think> and </think>. "
    "2. Tool Usage: During your reasoning, if you determine that you lack necessary information to answer accurately, "
    "you must call the search engine."
    " The search results will be returned to you between <tool_response> and </tool_response>. You may perform multiple searches if needed. "
    "3. Final Answer: After you have enough information, you must provide the final answer. "
    "When your reasoning confirms that you have sufficient information (whether you used the tool or not), provide the final answer. "
    "Enclose the final answer strictly between <answer> and </answer>. Do not include any additional explanations or reasoning in the final answer block."
    "For example, <answer> Beijing </answer>. Question: "
)

DEFAULT_USER_CONTENT_PREFIX = (
    "Follow this strict protocol to answer the question:\n"
    "1. Internal Reasoning: When you receive the question or new search results, you must first reason inside <think> and </think> . "
    "You can use both your own knowledge and the provided search results together, but do not fabricate facts. "
    "Base your judgment on overall evidence, not just fragments.\n"
    "2. Tool Usage: If you lack key information to answer reliably and clearly, you must call the search tool. "
    "Search results are provided between <tool_response> and </tool_response>. You can search multiple times if needed.\n"
    "3. Final Answer: When you have enough information (from knowledge or search), give the final answer. "
    "Wrap it strictly between <answer> and </answer> with no extra reasoning inside. "
    "Use appropriate confidence: definite only when supported strongly; use 'maybe' when evidence is partial.\n"
    "Example1:  <answer> yes </answer> .\n"
    "Example2:  <answer> maybe </answer> .\n"
    "Example3:  <answer> no </answer> .\n"
    "Question: "
)
def process_single_row(row, current_split_name, row_index):
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    question = row.get("question", "")
    # Build prompt structure
    user_content = user_content_prefix.rstrip("\n") + question
    prompt = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    # Extract ground truth from reward_model or fallback to golden_answers
    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("final_decision")
    else:
        ground_truth = row.get("final_decision", [])

    reward_model_data = {"ground_truth": ground_truth}
    reward_model_data["context"] = row.get("context")
    # Process data source
    data_source_tagged = "PubMedQA"

    # Build tools kwargs structure
    tools_kwargs = {
        "search": {
            "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}
        }
    }

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "agent_name": "tool_agent",
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
            "context": row.get("context"),
            "final_decision": row.get("final_decision"),
            "pubid": row.get("pubid"),
            "long_answer": row.get("long_answer"),
        }
    )


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    file_dir = '/nfsdata/yiao/PubMedQA/pqa_labeled/split'
    split = 'train'
    processed_files = []

    logger.info(f"Processing {split} split...")
    for file in os.listdir(file_dir):
        if file.endswith('.parquet') and split in file:
            try:
                file_path = os.path.join(file_dir, file)
                df_raw = pd.read_parquet(file_path)
                logger.info(f"Loaded {len(df_raw)} rows from {file_path}")

                def apply_process_row(row, split_name=split):
                        return process_single_row(row, current_split_name=split_name, row_index=row.name)

                df_processed = df_raw.apply(apply_process_row, axis=1)
                
                # Save processed DataFrame
                output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
                df_processed.to_parquet(output_file_path, index=False)
                logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
                processed_files.append(output_file_path)

            except Exception as e:
                logger.error(f"Error processing {split} split: {e}")
    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Search-R1 from HuggingFace, process, and save to Parquet.")
    parser.add_argument(
        "--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train", help="HuggingFace dataset repository ID."
    )
    parser.add_argument(
        "--local_dir",
        default="/nfsdata/yiao/PubMedQA/pqa_labeled/split/preprocessed_data",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")

    args = parser.parse_args()

    # System and user content configuration
    system_content = DEFAULT_SYSTEM_CONTENT
    user_content_prefix = DEFAULT_USER_CONTENT_PREFIX

    main()