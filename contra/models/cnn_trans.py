import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel
from setting import ADD_ATTN, BATCH_SIZE, CONTRA_WAY, CONTRASTIVE, DEVICE, LCM, ADD_DETAILS, ADD_CNN
from transformer import Transformer
from position_encoding import PostionalEncoding
import math
import torch.nn.functional as F
# from decoder import TransformerDecoder

class CNN_Encoder(nn.Module):
    def __init__(self, hid_dim=512, dropout=0.2, kernel_size=3, num_layers=5):
        super(CNN_Encoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.ModuleList([nn.Conv1d(hid_dim, hid_dim, kernel_size,
                                      padding=kernel_size // 2) for _ in range(num_layers)])
        self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)
    def forward(self, src):
        src = self.dropout(src)
        cnn = src.transpose(1, 2)
        for i, layer in enumerate(self.conv):
            cnn = F.tanh(layer(cnn)+cnn)        
        return cnn.transpose(1,2)

class CNN_Trans(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, max_length=512, maps=None, article_details=None, charge_details=None) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        self.article_class_num = len(maps["article2idx"])
        self.hid_dim = hid_dim
        self.max_length = max_length

        # for p in self.electra.parameters():
        #     p.requires_grad = False
        self.CNN = CNN_Encoder(hid_dim=hid_dim, dropout=0.2, kernel_size=3, num_layers=5)

        self.transformer_char_dec = Transformer(
                d_model=hid_dim, nhead=4, num_encoder_layers=1,num_decoder_layers=2, dim_feedforward=hid_dim * 8, dropout=0.1,
                activation="relu", normalize_before=False,
                return_intermediate_dec=False, 
                rm_self_attn_dec=True, rm_first_self_attn=True,)
        
        self.transformer_art_dec = Transformer(
                d_model=hid_dim, nhead=4, num_encoder_layers=1,num_decoder_layers=2, dim_feedforward=hid_dim * 8, dropout=0.1,
                activation="relu", normalize_before=False,
                return_intermediate_dec=False, 
                rm_self_attn_dec=True, rm_first_self_attn=True,)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # self.pos_embedding=nn.Embedding(max_length,hid_dim)
        self.pos_embedding = PostionalEncoding(hid_dim, max_len=max_length)
        
        self._register_detail_inputs(
            "article", article_details, self.article_class_num
        )
        self._register_detail_inputs(
            "charge", charge_details, self.charge_class_num
        )
        

        self.w2a=nn.Parameter(torch.zeros(hid_dim,hid_dim))
        self.w2c=nn.Parameter(torch.zeros(hid_dim,hid_dim))
        # Equation (2): y_hat_ij = sigmoid(W_ij^T h_i).
        # The paper does not add a predictor bias term.
        self.fc_article = GroupWiseLinear(len(maps["article2idx"]), hid_dim, bias=False)
        self.fc_charge = GroupWiseLinear(len(maps["charge2idx"]), hid_dim, bias=False)
        if ADD_DETAILS:
            self.transformer_enc = nn.TransformerEncoderLayer(emb_dim, nhead=8, batch_first=True)
            self.fill_article = nn.Parameter(
                torch.zeros(self.article_missing_indices.numel(), hid_dim)
            )
            self.fill_char = nn.Parameter(
                torch.zeros(self.charge_missing_indices.numel(), hid_dim)
            )
        else:
            self.label_emb_art = nn.Embedding(len(maps["article2idx"]), hid_dim)
            self.label_emb_char = nn.Embedding(len(maps["charge2idx"]), hid_dim)

    def _register_detail_inputs(self, name, tokenized_details, expected_count):
        input_ids = tokenized_details["input_ids"]
        if input_ids.shape[0] != expected_count:
            raise ValueError(
                f"{name} details must be aligned to {expected_count} labels, "
                f"got {input_ids.shape[0]}"
            )
        attention_mask = tokenized_details.get("attention_mask")
        if attention_mask is None:
            attention_mask = input_ids.ne(0).long()
        present_mask = tokenized_details.get("detail_present_mask")
        if present_mask is None:
            present_mask = torch.ones(expected_count, dtype=torch.bool)

        self.register_buffer(f"{name}_details", input_ids.clone())
        self.register_buffer(
            f"{name}_detail_attention_mask", attention_mask.clone().bool()
        )
        self.register_buffer(
            f"{name}_missing_indices",
            (~present_mask.bool()).nonzero(as_tuple=False).flatten(),
        )

    @staticmethod
    def _flatten_fact_inputs(text, attention_mask):
        if isinstance(text, (list, tuple)):
            if not text:
                raise ValueError("input_ids must contain at least one view")
            view_count = len(text)
            text = torch.cat(list(text), dim=0)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask] * view_count, dim=0)
        return text, attention_mask

    def details_attn(self, label_emb, another, enc_src, w, padding_mask=None):
        # seq_len = enc_src.shape[1]
        # enc_src, _ = torch.max(enc_src,dim=1)
        # enc_details = self.embedding(details)  
        # enc_details = self.transformer_enc(enc_details)
        # enc_details, _ = torch.max(enc_details, dim=1)  
        label_emb = label_emb[0] 
        scores = torch.matmul(torch.matmul(enc_src, w), label_emb.T)
        if padding_mask is not None:
            scores = scores.masked_fill(
                padding_mask.unsqueeze(-1), torch.finfo(scores.dtype).min
            )
        alpha = nn.Softmax(dim=1)(scores)
        context = torch.matmul(alpha, label_emb)

        # context=context.unsqueeze(1).repeat(1,seq_len,1)

        return context
    
    def details2label(self, batch_size, char_flag):
        if char_flag:
            details = self.charge_details
            attention_mask = self.charge_detail_attention_mask
            missing_indices = self.charge_missing_indices
            fill_values = self.fill_char
        else:
            details = self.article_details
            attention_mask = self.article_detail_attention_mask
            missing_indices = self.article_missing_indices
            fill_values = self.fill_article

        padding_mask = ~attention_mask
        enc_details = self.embedding(details)  
        enc_details = self.transformer_enc(
            enc_details, src_key_padding_mask=padding_mask
        )
        enc_details = enc_details.masked_fill(
            padding_mask.unsqueeze(-1), torch.finfo(enc_details.dtype).min
        )
        enc_details = torch.max(enc_details, dim=1).values
        if missing_indices.numel() > 0:
            enc_details = enc_details.index_copy(0, missing_indices, fill_values)
        label_emb = enc_details.unsqueeze(0).repeat(batch_size,1,1)
        return label_emb
    
    def label_enc(self, batch_size, char_flag):
        if char_flag:
            enc_details = self.label_emb_char.weight
        else:
            enc_details = self.label_emb_art.weight
        label_emb = enc_details.unsqueeze(0).repeat(batch_size,1,1)
        return label_emb
    
    def forward(self, data):
        fact_text = data["justice"]["input_ids"]
        attention_mask = data["justice"].get("attention_mask")
        fact_text, attention_mask = self._flatten_fact_inputs(
            fact_text, attention_mask
        )
        batch_size=fact_text.shape[0]
        padding_mask = (
            attention_mask.eq(0) if attention_mask is not None
            else fact_text.eq(0)
        )

        # enc_src = self.enc(fact_text)
        embedding = self.embedding(fact_text)
        enc_src = self.CNN(embedding)
        # char_context=char_context.unsqueeze(1).repeat(1,self.max_length,1) + enc_src
        # art_context=art_context.unsqueeze(1).repeat(1,self.max_length,1) + enc_src
        # char_context = torch.cat((enc_src, char_context), dim=-1)
        # art_context = torch.cat((enc_src, art_context), dim=-1)
        if ADD_DETAILS:
            char_label_emb = self.details2label(batch_size, 1)
            art_label_emb = self.details2label(batch_size, 0)
        else:
            char_label_emb = self.label_enc(batch_size, char_flag =1)
            art_label_emb = self.label_enc(batch_size, char_flag=0)
        
        if ADD_ATTN:
            char_context = self.details_attn(label_emb=char_label_emb, another=art_label_emb, 
                                            enc_src = enc_src, w = self.w2c,
                                            padding_mask=padding_mask)
            art_context = self.details_attn(label_emb=art_label_emb, another=char_label_emb,
                                            enc_src = enc_src, w = self.w2a,
                                            padding_mask=padding_mask)
            char_context = char_context + enc_src   
            art_context = art_context + enc_src
        else:
            char_context, art_context = enc_src, enc_src
    
        # pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).to(DEVICE)
        # pos_emb = self.pos_embedding(pos).to(DEVICE)
        pos_emb = self.pos_embedding(fact_text)

        char_hs = self.transformer_char_dec(
            char_context, char_label_emb, pos_emb, mask=padding_mask
        )[0] # B,K,d
        art_hs = self.transformer_art_dec(
            art_context, art_label_emb, pos_emb, mask=padding_mask
        )[0]
        out_charge = self.fc_charge(char_hs[-1])
        out_article = self.fc_article(art_hs[-1])

        return {
            "article": out_article,
            "charge": out_charge,
            "char_enc": char_hs,
            "art_enc": art_hs
            # "penalty": out_penalty
        }
    
class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        logits = (self.W * x).sum(-1)
        if self.bias:
            logits = logits + self.b
        return torch.sigmoid(logits)

class LCM_DIST(nn.Module):
    def __init__(self, output_dim, hid_dim=512, wvdim=256) -> None:
        super().__init__()
        # label_encoder:
        self.label_emb = nn.Embedding(output_dim,wvdim) # (n,wvdim)
        self.label_fc = nn.Linear(wvdim, hid_dim)

        self.sim_fc = nn.Linear(output_dim, output_dim)

    def forward(self, embedded, labels):
        # print(text_emb.shape,'text')  # [16,64]
        label_emb = self.label_emb(labels)
        label_emb = F.tanh(self.label_fc(label_emb))
        # print(label_emb.shape,'label')  # [16,20,64]
        doc_product = torch.bmm(label_emb, embedded.unsqueeze(-1))  # (b,n,d) dot (b,d,1) --> (b,n,1)
        # print(doc_product.shape)   # [16,20,1]
        label_sim_dict = self.sim_fc(doc_product.squeeze(-1))
        #print(label_sim_dict.shape)
        return label_sim_dict

class ProjectionHead(nn.Module):
    def __init__(self, output_dim, output_art_dim, feat_dim=128):
        super(ProjectionHead, self).__init__()
        # label_encoder:
        self.head = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, feat_dim)
            )
        self.head_art = nn.Sequential(
                nn.Linear(output_art_dim, output_art_dim),
                nn.ReLU(inplace=True),
                nn.Linear(output_art_dim, feat_dim)
            )

    def forward(self, output, output_art):
        output_cl = self.head(output)
        output_art_cl = self.head_art(output_art)
        #print(label_sim_dict.shape)
        return output_cl, output_art_cl
