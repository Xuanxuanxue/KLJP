from pathlib import Path
import os

import torch

from runtime import build_runtime_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME = build_runtime_config(PROJECT_ROOT)
RUNTIME_MODE = RUNTIME.mode
RUNTIME_MODE_REASON = RUNTIME.reason
DATA_ROOT = RUNTIME.data_root
LOG_ROOT = RUNTIME.log_root
DATASET_ROOT = DATA_ROOT / "dataset"
LABEL2ID_ROOT = DATA_ROOT / "label2id"
LABEL2LABEL_ROOT = DATA_ROOT / "label2label"
LABEL_DETAILS_ROOT = DATA_ROOT / "label_details"
CACHE_ROOT = DATA_ROOT / "cache"

TRAIN_FILE = DATASET_ROOT / "train.json"
DEV_FILE = DATASET_ROOT / "valid.json"
TEST_FILE = DATASET_ROOT / "test.json"

# Only used by the non-pretrained baselines.
EMBEDDING_PATH = RUNTIME.embedding_path

BERT_MODEL_NAME = "bert-base-chinese"
ELECTRA_MODEL_NAME = "hfl/chinese-electra-180g-small-discriminator"
LOCAL_ELECTRA_MODEL_PATH = RUNTIME.electra_path
ALLOW_MODEL_DOWNLOAD = RUNTIME.allow_model_download
if not ALLOW_MODEL_DOWNLOAD:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if RUNTIME_MODE == "server":
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))

MODEL = "Al_Trans"
# TextCNN,LSTM,Transformer,TopJudge,Electra,Al_Trans,Attention_XML,CNN_Trans
if MODEL=="Electra" or MODEL=="Al_Trans" or MODEL=="Bert":
    PRETRAIN = True 
else:
    PRETRAIN = False 

ADD_DETAILS = True
HEAD = False
ADD_ATTN = False
ADD_CNN = False

K_CONS = False
SMOOTH = False
BEAM = False
LCM = False
CONTRASTIVE = False
CONTRA_WAY = "supcon2" #"r-drop","supcon2","supcon"

# Static GJudge confusion-graph augmentation.  It is off by default so E0
# remains the original K-LJP model.  The two per-task flags make E1/E2/E3
# ablations possible without touching the static A2C/C2A mappings.
USE_CONFUSION_GRAPH = os.environ.get("KLJP_USE_CONFUSION_GRAPH", "0").lower() in {
    "1", "true", "yes", "on"
}
USE_ARTICLE_CONFUSION_GRAPH = os.environ.get(
    "KLJP_USE_ARTICLE_CONFUSION_GRAPH",
    "1" if USE_CONFUSION_GRAPH else "0",
).lower() in {"1", "true", "yes", "on"}
USE_CHARGE_CONFUSION_GRAPH = os.environ.get(
    "KLJP_USE_CHARGE_CONFUSION_GRAPH",
    "1" if USE_CONFUSION_GRAPH else "0",
).lower() in {"1", "true", "yes", "on"}
CONFUSION_GRAPH_THRESHOLD = float(
    os.environ.get("KLJP_CONFUSION_GRAPH_THRESHOLD", "0.80")
)
CONFUSION_GRAPH_WEIGHT = float(
    os.environ.get("KLJP_CONFUSION_GRAPH_WEIGHT", "1.0")
)
CONFUSION_GRAPH_HEADS = int(os.environ.get("KLJP_CONFUSION_GRAPH_HEADS", "4"))
CONFUSION_GRAPH_DROPOUT = float(
    os.environ.get("KLJP_CONFUSION_GRAPH_DROPOUT", "0.10")
)
CONFUSION_GRAPH_TOPK = int(os.environ.get("KLJP_CONFUSION_GRAPH_TOPK", "8"))
ARTICLE_GRAPH_ALPHA = float(
    os.environ.get(
        "KLJP_ARTICLE_GRAPH_ALPHA",
        "0.1" if USE_ARTICLE_CONFUSION_GRAPH else "0.0",
    )
)
CHARGE_GRAPH_ALPHA = float(
    os.environ.get(
        "KLJP_CHARGE_GRAPH_ALPHA",
        "0.1" if USE_CHARGE_CONFUSION_GRAPH else "0.0",
    )
)
LAMBDA_GRAPH = float(os.environ.get("KLJP_LAMBDA_GRAPH", "0.0"))
if ARTICLE_GRAPH_ALPHA < 0 or CHARGE_GRAPH_ALPHA < 0 or LAMBDA_GRAPH < 0:
    raise ValueError("graph alpha and lambda_graph must be non-negative")

DEVICE = torch.device(
    os.environ.get(
        "KLJP_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
    )
)
EMB_DIM = 256
HID_DIM = 256
SEQ_LEN = 512
BATCH_SIZE = int(
    os.environ.get(
        "KLJP_BATCH_SIZE",
        16 if RUNTIME_MODE == "server" and PRETRAIN else
        128 if RUNTIME_MODE == "server" else
        8 if PRETRAIN else 64,
    )
)
NUM_WORKERS = int(
    os.environ.get("KLJP_NUM_WORKERS", 4 if RUNTIME_MODE == "server" else 0)
)
PIN_MEMORY = DEVICE.type == "cuda"
EPOCHS = 16 if PRETRAIN else 100
