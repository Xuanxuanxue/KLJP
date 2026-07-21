from torch.utils.data import Dataset
import torch
from functools import lru_cache
import jieba
import re
import collections
from transformers import BertTokenizer, AutoTokenizer
from tokenizer import MyTokenizer
from pretrained_loader import load_electra_tokenizer
import json
from setting import (BERT_MODEL_NAME, CONTRA_WAY, CONTRASTIVE,
                     DATA_ROOT, ELECTRA_MODEL_NAME, LABEL2ID_ROOT,
                     LABEL2LABEL_ROOT, DEVICE, MODEL)


RANDOM_SEED = 19
torch.manual_seed(RANDOM_SEED)

def text_cleaner(text):
    def load_stopwords(filename):
        stopwords = []
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.replace("\n", "")
                stopwords.append(line)
        return stopwords

    stop_words = load_stopwords(DATA_ROOT / "stopwords.txt")

    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        # newline after </p> and </div> and <h1/>...
        {r'</(div)\s*>\s*': u'\n'},
        # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        # show links instead of texts
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]

    # 替换html特殊字符
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace(
        "&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    text = text.replace('+', ' ').replace(',', ' ').replace(':', ' ')
    text = re.sub("([0-9]+[年月日])+", "", text)
    text = re.sub("[a-zA-Z]+", "", text)
    text = re.sub(r"[0-9.]+元", "", text)
    stop_words_user = ["年", "月", "日", "时", "分", "许", "某", "甲", "乙", "丙"]
    word_tokens = jieba.cut(text)

    def str_find_list(string, words):
        for word in words:
            if string.find(word) != -1:
                return True
        return False

    text = [w for w in word_tokens if w not in stop_words if not str_find_list(w, stop_words_user)
            if len(w) >= 1 if not w.isspace()]
    return " ".join(text)

#只清洗数据，不去停用词。
def text_cleaner2(text):
    # 替换html特殊字符
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace(
        "&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    # 换行替换为#, 空格替换为&
    text = text.replace("#", "").replace("$", "").replace("&", "")
    text = text.replace("\n", "").replace(" ", "")

    return text

@lru_cache(maxsize=None)
def _get_tokenizer(model, embedding_path):
    if model == "Bert":
        return BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    if model in ("Electra", "Al_Trans"):
        return load_electra_tokenizer(AutoTokenizer.from_pretrained)
    return MyTokenizer(embedding_path)


class LazyLegalDataset(Dataset):
    """Read JSONL records on demand and tokenize one DataLoader batch at a time."""

    def __init__(self, filename, seq_len, tokenizer, maps, text_clean=False):
        self.filename = str(filename)
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.maps = maps
        self.text_clean = text_clean
        self._file = None
        self.offsets = []

        with open(self.filename, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def _open_file(self):
        if self._file is None or self._file.closed:
            self._file = open(self.filename, "rb")
        return self._file

    def __getitem__(self, idx):
        f = self._open_file()
        f.seek(self.offsets[idx])
        record = json.loads(f.readline())
        fact = record["fact"]
        if self.text_clean:
            fact = text_cleaner(fact)
        return {
            "fact": fact,
            "charge_indices": [self.maps["charge2idx"][x]
                               for x in record["meta"]["accusation"]],
            "article_indices": [self.maps["article2idx"][str(x)]
                                for x in record["meta"]["relevant_articles"]],
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def collate_fn(self, samples):
        justice = self.tokenizer(
            [sample["fact"] for sample in samples],
            return_tensors="pt",
            padding="max_length",
            max_length=self.seq_len,
            truncation=True,
        )
        if "token_type_ids" not in justice:
            justice["token_type_ids"] = torch.zeros_like(justice["input_ids"])

        charge = torch.zeros(len(samples), len(self.maps["charge2idx"]), dtype=torch.long)
        article = torch.zeros(len(samples), len(self.maps["article2idx"]), dtype=torch.long)
        for row, sample in enumerate(samples):
            charge[row, sample["charge_indices"]] = 1
            article[row, sample["article_indices"]] = 1

        input_ids = justice["input_ids"]
        if CONTRASTIVE and CONTRA_WAY == "supcon2":
            input_ids = [input_ids, input_ids.clone()]

        batch_size = len(samples)
        return {
            "justice": {
                "input_ids": input_ids,
                "token_type_ids": justice["token_type_ids"],
                "attention_mask": justice["attention_mask"],
            },
            "charge": charge,
            "charge_num": torch.tensor(
                [len(sample["charge_indices"]) - 1 for sample in samples]
            ),
            "charge_label": torch.arange(len(self.maps["charge2idx"])).repeat(batch_size, 1),
            "article": article,
            "article_num": torch.tensor(
                [len(sample["article_indices"]) - 1 for sample in samples]
            ),
            "article_label": torch.arange(len(self.maps["article2idx"])).repeat(batch_size, 1),
        }

def simple_load_multi_data(filename, seq_len, embedding_path, text_clean:bool):
    #读取字典
    maps = {}
    c2i_path = LABEL2ID_ROOT / "c2i_clean.json"
    a2i_path = LABEL2ID_ROOT / "a2i_clean.json"
    with open(c2i_path) as f:
        c2i = json.load(f)
        maps["charge2idx"] = c2i
        maps["idx2charge"] = {str(v): k for k, v in c2i.items()}

    with open(a2i_path) as f:
        a2i = json.load(f)
        maps["article2idx"] = a2i
        maps["idx2article"] = {str(v): k for k, v in a2i.items()}

#读取对应关系，并把它们转换为id
    with open(LABEL2LABEL_ROOT / "c2a_clean.json") as f:
        c2a_final = json.load(f)
        c2a_tmp = dict()
        for key, value in c2a_final.items():
            # KLJP-DATA stores each related article in a one-element list.
            if isinstance(value, list):
                value = value[0]
            c2a_tmp[maps["charge2idx"][key]] = maps["article2idx"][value]
        maps["c2a"]=c2a_tmp
    dic = collections.defaultdict(list)
    for k,v in maps["c2a"].items():
        dic[v].append(k)
    maps["a2c"] = dic

    tokenizer = _get_tokenizer(MODEL, str(embedding_path))
    dataset = LazyLegalDataset(filename, seq_len, tokenizer, maps, text_clean)

    return dataset, maps

def load_details(maps, art_details_path, char_details_path, art_len, char_len, embedding_path):
    with open(art_details_path,'r',encoding="utf-8") as f:
        article_detail_map=json.load(f)
    with open(char_details_path,'r',encoding="utf-8") as f:
        charge_detail_map=json.load(f)

    def align_details(label2idx, detail_map, text_getter):
        """Return one detail per label, ordered by the model's label id."""
        label_count = len(label2idx)
        details = [""] * label_count
        present_mask = torch.zeros(label_count, dtype=torch.bool)
        for label, label_id in label2idx.items():
            raw_detail = detail_map.get(label)
            if raw_detail is None:
                continue
            detail = text_getter(raw_detail).replace(" ", "").strip()
            if detail:
                details[label_id] = detail
                present_mask[label_id] = True
        return details, present_mask

    article_details, article_present_mask = align_details(
        maps["article2idx"], article_detail_map, lambda value: value
    )
    charge_details, charge_present_mask = align_details(
        maps["charge2idx"], charge_detail_map, lambda value: value["定义"]
    )

    if MODEL=="Bert":
        tokenizer=BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    elif MODEL=="Electra" or MODEL=="Al_Trans":
        tokenizer=load_electra_tokenizer(AutoTokenizer.from_pretrained)
    else:
        tokenizer=MyTokenizer(embedding_path)
    article_details = tokenizer(article_details, return_tensors="pt",
                        padding="max_length", max_length=art_len, truncation=True)
    charge_details = tokenizer(charge_details, return_tensors="pt",
                        padding="max_length", max_length=char_len, truncation=True)
    article_details["detail_present_mask"] = article_present_mask
    charge_details["detail_present_mask"] = charge_present_mask

    return article_details, charge_details

def tocuda(data):
    if type(data) is dict:
        for k in data:
            data[k] = data[k].to(DEVICE)
    else:
        data = data.to(DEVICE)
    return data
