import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelBase(nn.Module):
    def __init__(
        self, 
        # ========== ADD: args 추가, 나머지 삭제 =========
        args,
        # ============================================
    ):
        super().__init__()
        # =========== ADD: args 불러오기 ================
        self.args = args
        # =============================================

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # ============== ADD:  속성 추가 ================
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.device = self.args.device
        # =============================================

        # ========= ADD: args 중 n_xx 불러오기 ===========
        # setattr(x, 'y', v) is equivalent to 'x.y = v'
        self.n_args = [arg for arg in vars(self.args) if arg.startswith('n_')]
        for arg in self.n_args:
            value = getattr(self.args, arg)
            setattr(self, arg, value) 
        # =============================================

        # ============= ADD: Embeddings ===============
        # getattr(x, 'y') is equivalent to x.y.
        hd, intd = self.hidden_dim, self.hidden_dim // self.args.dim_div
        self.embedding_interaction = nn.Embedding(3, intd)   
        for cate_col in self.args.cate_cols:
            n = getattr(self, f'n_{cate_col}') # n = self.n_xx 의 값 
            setattr(self, f'embedding_{cate_col}', nn.Embedding(n + 1, intd)) # self.embedding_xx = nn.Embedding(n + 1, intd)
        # ==============================================

        # ============ ADD: Concat Projection ==========
        self.comb_proj = nn.Sequential(
            nn.Linear(intd * (len(self.args.cate_cols) + 1), hd // 2), # cate_cols 개수 + interaction (categorical)
            nn.LayerNorm(hd // 2)
        )
        self.cont_proj = nn.Sequential(
            nn.Linear(len(self.args.cont_cols), hd // 2), # cont_cols 개수
            nn.LayerNorm(hd // 2)
        )
        # ==============================================

        self.fc = nn.Linear(hd, 1)

    # =========== ADD: data ===============
    def forward(self, data): 
    # =====================================

        # ============== ADD: 데이터 길이 확인 =========================
        len_check = len(self.args.cate_cols + self.args.cont_cols + self.args.target_cols) + 2
        assert len(data) == len_check, f'실제 데이터 길이는 {len(data)} 인데, args로 넘겨받은 총 길이는 {len_check}'
        # ==========================================================
        interaction = data[-1]
        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction.int())
        # ================== ADD: embed_xx ===================
        embed_cate_feats = []

        for cate_col, value in zip(self.args.cate_cols, data[:len(self.args.cate_cols)]):
            embed_cate_feat = getattr(self, f'embedding_{cate_col}')(value.int()) # self.embedding_xxx(xxx.int())
            embed_cate_feats.append(embed_cate_feat)

        embed = torch.cat(([embed_interaction, *embed_cate_feats]), dim=2)
        # =====================================================

        # ======== ADD: Concatenate continous feature =========
        # TODO: (중요) data 길이 vs cate_cols 길이 확인 필요
        cont_feats = []
        for cont_col, value in zip(self.args.cont_cols, data[len(self.args.cate_cols):-3]):
            cont_feats.append(value.unsqueeze(2))
        cont_features = torch.cat(cont_feats, dim=2).float()
        # =====================================================

        # =========== ADD: Projection =========================
        cate = self.comb_proj(embed)
        cont = self.cont_proj(cont_features)
        X = torch.cat([cate, cont], dim=2)
        # =====================================================

# LastQuery 구현 
class Feed_Forward_block(nn.Module):
    """
    out =  GELU( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.gelu(self.layer1(ffn_in)))


class LastQuery(ModelBase):
    """
    embedding --> Multihead Attention --> LSTM
    """
    def __init__(
        self,
        # ======= ADD: args 추가 ========
        args,
        # ==============================
    ):
        super().__init__(
            # ==== ADD: ModelBase에서 args 불러오기 ====
            args
            # =======================================
        )
        
        # === Encoder 
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.n_heads)
        self.mask = None # not used
        self.ffn = Feed_Forward_block(self.hidden_dim)

        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)

        # === LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.activation = nn.Sigmoid()


    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device) # ====== ADD: self.device
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device) # ====== ADD: self.device
        return (h, c)

    def forward(self, data):
        # ================== ADD: mask 불러오기 ======================
        # 순서: self.args.cate_cols + self.args.cont_cols + self.args.target_cols + mask + interaction
        mask = data[-2]
        interaction = data[-1]
        seq_len = interaction.size(1)
        # ==========================================================

        # Embedding
        X, batch_size = super().forward(data)
        
        # Encoder 
        # make attention_mask (batch_size * n_head, seq_len)
        attention_mask = mask.repeat(1, self.n_heads)
        attention_mask = attention_mask.view(batch_size * self.n_heads, -1, seq_len)
        attention_mask = (1.0 - attention_mask) * -10000.0
        head_mask = [None] * self.n_layers # not used 

        # Attention
        q = self.query(X)[:, -1:, :].permute(1, 0, 2) # last query only
        k = self.key(X).permute(1, 0, 2)
        v = self.value(X).permute(1, 0, 2)
        out, _ = self.attn(q, k, v, attn_mask=attention_mask)
        
        out = out.permute(1, 0, 2)
        out = X + out               # Residual connection
        out = self.layer_norm1(out) # Layer normalization

        # Feed Forward Network
        out = self.ffn(out)
        out = X + out               # Residual connection
        out = self.layer_norm2(out) # Layer normalization

        # LSTM
        hidden = self.init_hidden(batch_size) # (h, c)
        out, hidden = self.lstm(out, hidden)

        # DNN 
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out) # fully connected

        # Activation Function
        preds = self.activation(out).view(batch_size, -1) 

        return preds