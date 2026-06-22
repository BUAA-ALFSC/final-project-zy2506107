from implementation import config as fs_config
from implementation import funsearch

import bin_packing_utils
import funsearch_bin_packing_local_llm as base


if __name__ == "__main__":
    class_config = fs_config.ClassConfig(
        llm_class=base.LocalLLM,
        sandbox_class=base.Sandbox,
    )
    experiment_config = fs_config.Config(
        samples_per_prompt=1,
        evaluate_timeout_seconds=30,
    )
    inputs = {"OR3": bin_packing_utils.datasets["OR3"]}
    funsearch.main(
        specification=base.specification,
        inputs=inputs,
        config=experiment_config,
        max_sample_nums=40,
        class_config=class_config,
        log_dir="logs/funsearch_qwen_real_formal",
    )
