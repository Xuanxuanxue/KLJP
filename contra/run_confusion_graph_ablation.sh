#!/usr/bin/env bash

# 混淆图消融实验脚本
#
# 默认运行以下 5 组实验：
#   1. baseline：不启用法条图和罪名图
#   2. article_graph：只启用法条混淆图
#   3. charge_graph：只启用罪名混淆图
#   4. both_graph：同时启用两张图
#   5. both_graph_loss：两张图开启，并加入图损失
#
# 使用示例：
#   bash contra/run_confusion_graph_ablation.sh
#   KLJP_EXPERIMENT=5 bash contra/run_confusion_graph_ablation.sh
#   KLJP_SEEDS="22 67" bash contra/run_confusion_graph_ablation.sh
#   KLJP_CONFUSION_GRAPH_TOPK=5 bash contra/run_confusion_graph_ablation.sh
#
# 也可以指定 Python 解释器：
#   PYTHON_BIN=/root/miniconda3/envs/kljp/bin/python \
#   bash contra/run_confusion_graph_ablation.sh

set -euo pipefail

# 当前脚本所在目录。这样从项目根目录或其他目录执行都不会丢失路径。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_bl.py"

# 可通过环境变量覆盖 Python 解释器；默认使用 PATH 中的 python。
PYTHON_BIN="${PYTHON_BIN:-python}"

# 默认只跑一个随机种子，便于快速检查配置。
# 多随机种子示例：KLJP_SEEDS="22 67 81"。
SEEDS="${KLJP_SEEDS:-22}"

# 选择实验编号：all、1、2、3、4 或 5。
# 默认 all；本次只做实验 5 时设置 KLJP_EXPERIMENT=5。
EXPERIMENT="${KLJP_EXPERIMENT:-all}"
case "${EXPERIMENT}" in
    all|1|2|3|4|5)
        ;;
    *)
        echo "错误：KLJP_EXPERIMENT 必须是 all、1、2、3、4 或 5。" >&2
        exit 1
        ;;
esac

# 法条图默认最多保留每个节点的 8 个邻居。
# 可测试 3、5、8：KLJP_CONFUSION_GRAPH_TOPK=3/5/8。
TOPK="${KLJP_CONFUSION_GRAPH_TOPK:-8}"

run_one_experiment() {
    local experiment_name="$1"
    local article_graph="$2"
    local charge_graph="$3"
    local article_alpha="$4"
    local charge_alpha="$5"
    local lambda_graph="$6"
    local seed="$7"

    echo "============================================================"
    echo "开始实验: ${experiment_name} | seed=${seed}"
    echo "article_alpha=${article_alpha} | charge_alpha=${charge_alpha}"
    echo "lambda_graph=${lambda_graph} | topk=${TOPK}"
    echo "============================================================"

    # 每组实验都显式设置所有图相关开关，避免继承外部环境造成配置污染。
    env \
        KLJP_USE_ARTICLE_CONFUSION_GRAPH="${article_graph}" \
        KLJP_USE_CHARGE_CONFUSION_GRAPH="${charge_graph}" \
        KLJP_ARTICLE_GRAPH_ALPHA="${article_alpha}" \
        KLJP_CHARGE_GRAPH_ALPHA="${charge_alpha}" \
        KLJP_LAMBDA_GRAPH="${lambda_graph}" \
        KLJP_CONFUSION_GRAPH_TOPK="${TOPK}" \
        "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
        --seed "${seed}"
}

run_selected_experiment() {
    local experiment_id="$1"
    shift

    # 只有选择 all 或对应编号时才执行该实验。
    if [[ "${EXPERIMENT}" == "all" || "${EXPERIMENT}" == "${experiment_id}" ]]; then
        run_one_experiment "$@"
    fi
}

for seed in ${SEEDS}; do
    # 实验 1：baseline。
    # alpha=0 且 lambda_graph=0 时，模型 forward 和损失均不受图模块影响。
    run_selected_experiment 1 "baseline" 0 0 0.0 0.0 0.0 "${seed}"

    # 实验 2：只启用法条混淆图。
    run_selected_experiment 2 "article_graph" 1 0 0.1 0.0 0.0 "${seed}"

    # 实验 3：只启用罪名混淆图。
    run_selected_experiment 3 "charge_graph" 0 1 0.0 0.1 0.0 "${seed}"

    # 实验 4：同时启用法条图和罪名图。
    run_selected_experiment 4 "both_graph" 1 1 0.1 0.1 0.0 "${seed}"

    # 实验 5：两张图开启，并加入可选图损失。
    run_selected_experiment 5 "both_graph_loss" 1 1 0.1 0.1 0.01 "${seed}"
done

echo "全部混淆图消融实验完成。"
