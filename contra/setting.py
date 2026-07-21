from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "KLJP-DATA"
# Training artifacts are stored in the WSL home directory. This corresponds to
# \\wsl.localhost\Ubuntu-20.04\home\user\KLJP-logs in Windows Explorer.
LOG_ROOT = Path("/home/user/KLJP-logs")
DATASET_ROOT = DATA_ROOT / "dataset"
LABEL2ID_ROOT = DATA_ROOT / "label2id"
LABEL2LABEL_ROOT = DATA_ROOT / "label2label"
LABEL_DETAILS_ROOT = DATA_ROOT / "label_details"
CACHE_ROOT = DATA_ROOT / "cache"

TRAIN_FILE = DATASET_ROOT / "train.json"
DEV_FILE = DATASET_ROOT / "valid.json"
TEST_FILE = DATASET_ROOT / "test.json"

# Only used by the non-pretrained baselines.
EMBEDDING_PATH = PROJECT_ROOT / "gensim_train" / "word2vec.model"

BERT_MODEL_NAME = "bert-base-chinese"
ELECTRA_MODEL_NAME = "hfl/chinese-electra-180g-small-discriminator"
LOCAL_ELECTRA_MODEL_PATH = (
    PROJECT_ROOT.parent
    / "kljp2"
    / "chinese-electra-180g-small-discriminator"
)

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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMB_DIM = 256
HID_DIM = 256
SEQ_LEN = 512
BATCH_SIZE = 16 if PRETRAIN else 128
EPOCHS = 16 if PRETRAIN else 100
