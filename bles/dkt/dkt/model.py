import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel
import torch.nn.functional as F
import numpy as np

class ModelBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        ## add (기존 범주 수에 + 1)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags
        ## add

        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = hidden_dim, hidden_dim // 3
        self.embedding_interaction = nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)

        # Concatentaed Embedding Projection
        ## add
        self.comb_proj = nn.Sequential(
            nn.Linear(intd * 4, hd // 2), # 범주형 컬럼 개수: 4
            nn.LayerNorm(hd //2)
        ) 
            
        ## add
        self.cont_proj = nn.Sequential(
            nn.Linear(1, hd // 2), # 연속형 컬럼 개수: 1
            nn.LayerNorm(hd // 2)
        )

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)
    ## add
    def forward(self, test, question, tag, correct, mask, interaction, elapsed):
        batch_size = interaction.size(0)
        # Embedding
        ## add
        embed_interaction = self.embedding_interaction(interaction.int())
        embed_test = self.embedding_test(test.int())
        embed_question = self.embedding_question(question.int())
        embed_tag = self.embedding_tag(tag.int())
        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            dim=2,
        )
        ## add
        elapsed = elapsed.unsqueeze(2)
        
        # cont_features = torch.cat([elapsed, continuous_tag, user_elapsed_answerCode], dim=2)
        cont_features = torch.cat([elapsed], dim=2)
        cont_features = cont_features.float()
        
        ## add
        cate = self.comb_proj(embed)
        cont = self.cont_proj(cont_features)
        X = torch.cat([cate, cont], dim=2)
        
        return X, batch_size


class LGCNModelBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        ## add (기존 범주 수에 + 1)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags
        ## add

        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = hidden_dim, hidden_dim // 3
        self.embedding_interaction = nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)

        # Concatentaed Embedding Projection
        ## add
        self.comb_proj = nn.Sequential(
            nn.Linear(intd * 7, hd // 2), # 범주형 컬럼 개수: 7
            nn.LayerNorm(hd //2)
        ) 
            
        ## add
        self.cont_proj = nn.Sequential(
            nn.Linear(1, hd // 2), # 연속형 컬럼 개수: 1
            nn.LayerNorm(hd // 2)
        )

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)
        
        # LightGCN Embedding
        self.graph_embedding_question = np.load('/opt/ml/input/code/lightgcn/embedding/embedding_assessmentItemID.npy')
        self.graph_embedding_test = np.load('/opt/ml/input/code/lightgcn/embedding/embedding_testId.npy')
        self.graph_embedding_tag = np.load('/opt/ml/input/code/lightgcn/embedding/embedding_KnowledgeTag.npy')
        self.n_users = 7442
        
        # graph 모델에서 지정했던 hidden_dim으로 맞춰줘야 함
        self.graph_linear_question = nn.Linear(64, intd)
        self.graph_linear_test = nn.Linear(64, intd)
        self.graph_linear_tag = nn.Linear(64, intd)
        
    ## add
    def forward(self, test, question, tag, correct, mask, interaction, elapsed):
        batch_size = interaction.size(0)
        # Embedding
        ## add
        embed_interaction = self.embedding_interaction(interaction.int())
        embed_test = self.embedding_test(test.int())
        
        ##### graph_embed_question
        embed_question = self.embedding_question(question.int())
        
        question_ = question.detach().cpu().numpy()
        
        graph_embed_question = []
        for user in question_:
            users = []
            for i in user:
                users.append(self.graph_embedding_question[self.n_users - 1 + i])
            graph_embed_question.append(users)

        graph_embed_question = torch.Tensor(np.array(graph_embed_question)).to('cuda')
                
        graph_embed_question = self.graph_linear_question(graph_embed_question)
        
        
        ####
        
        embed_test = self.embedding_test(test.int())
        
        test_ = test.detach().cpu().numpy()
        
        graph_embed_test = []
        for user in test_:
            users = []
            for i in user:
                users.append(self.graph_embedding_test[self.n_users - 1 + i])
            graph_embed_test.append(users)

        graph_embed_test = torch.Tensor(np.array(graph_embed_test)).to('cuda')
                
        graph_embed_test = self.graph_linear_test(graph_embed_test)
        
        #####
        
        embed_tag = self.embedding_tag(tag.int())
        
        tag_ = tag.detach().cpu().numpy()
        
        graph_embed_tag = []
        for user in tag_:
            users = []
            for i in user:
                users.append(self.graph_embedding_tag[self.n_users - 1 + i])
            graph_embed_tag.append(users)

        graph_embed_tag = torch.Tensor(np.array(graph_embed_tag)).to('cuda')
                
        graph_embed_tag = self.graph_linear_tag(graph_embed_tag)
        
        
        #####
        
        embed_tag = self.embedding_tag(tag.int())
        
        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                graph_embed_question,
                graph_embed_test,
                graph_embed_tag,
            ],
            dim=2,
        )
        ## add
        elapsed = elapsed.unsqueeze(2)
        
        # cont_features = torch.cat([elapsed, continuous_tag, user_elapsed_answerCode], dim=2)
        cont_features = torch.cat([elapsed], dim=2)
        cont_features = cont_features.float()
        
        ## add
        cate = self.comb_proj(embed)
        cont = self.cont_proj(cont_features)
        X = torch.cat([cate, cont], dim=2)
        
        return X, batch_size


class LSTM(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags,
        )
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

    ## add
    def forward(self, test, question, tag, correct, mask, interaction, elapsed):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        elapsed=elapsed,
                                        )
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,

        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags,
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            # Feed-Forward-Network 안에서의 Hidden dimension
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

    ## add
    def forward(self, test, question, tag, correct, mask, interaction, elapsed):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        elapsed=elapsed,
                                        )

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        # print(f"mask.shape:{mask.shape}") # [64, 20]
        # print(mask)
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # print(f"extended_attention_mask.shape:{extended_attention_mask.shape}") # [64, 1, 1, 20]
        head_mask = [None] * self.n_layers
        # extended_attention_mask: 사용할 값은 0으로, padding은 -100000으로 mask 생성해서
        # attention 매커니즘에 활용
        # mask: 1 -> attention에 영향을 미치지 않기 위해 0으로
        # mask: 0 -> 해당 시점에서의 정보를 사용하지 않기 위해 -10000으로
        # print(extended_attention_mask)
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class BERT(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags,
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=max_seq_len,
        )
        self.encoder = BertModel(self.config)

    ## add
    def forward(self, test, question, tag, correct, mask, interaction, elapsed):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        elapsed=elapsed,
                                        )

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out

class GRU(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags,
        )
        self.gru = nn.GRU(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

    ## add
    def forward(self, test, question, tag, correct, mask, interaction, elapsed):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        elapsed=elapsed,
                                        )
        out, _ = self.gru(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class GRUATTN(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags,
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.gru = nn.GRU(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

    ## add
    def forward(self, test, question, tag, correct, mask, interaction, elapsed):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        elapsed=elapsed,
                                        )
        # print(test)
        # print(test.size())
        out, _ = self.gru(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        # print(f"mask.shape:{mask.shape}") # [64, 20]
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # print(f"extended_attention_mask.shape:{extended_attention_mask.shape}") # [64, 1, 1, 20]
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out
    
class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
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
        hidden_dim: int = 64,
        n_layers: int = 1,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 100,
        **kwargs    
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags,
        )
        self.n_heads = n_heads
        self.drop_out = drop_out

        # === Embedding (ModelBase)
        # hd: hidden dimension
        # intd: intermediate hidden dimension
        # hd, intd = hidden_dim, hidden_dim // 3

        # embedding_interaction (3 -> intd)
        # embedding_test        (n_tests + 1 -> intd)
        # embedding_question    (n_questions + 1 -> intd)
        # embedding_tag         (n_tags + 1 -> intd)
        # comb_proj             (intd * (len(features) + 1) -> hd) # Concatenated Embedding Projection

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

        # === Fully Connected Layer 
        # fc (hd -> 1)

        self.activation = nn.Sigmoid()


    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to("cuda")
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to("cuda")
        return (h, c)

    def forward(self, test, question, tag, correct, mask, interaction, elapsed):
        # print(f'interaction size: {interaction.size()}')
        seq_len = interaction.size(1)
        # print(f'seq_len : {seq_len}')
        # print(f'mask: {mask.size()}')

        # Embedding
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        elapsed=elapsed,
                                        )
        
        # TODO: Add new features
        # 일단은 ModelBase 에서 처리하는 방향으로 진행 예정
        # embed_timebin = self.embedding_timebin(timebin.int().to('cuda'))
        # X = torch.cat([X, embed_timebin.unsqueeze(1)], dim=2)
        
        # Encoder 
        # make attention_mask (size: batch_size * n_head, seq_len)
        attention_mask = mask.repeat(1, self.n_heads)
        attention_mask = attention_mask.view(batch_size * self.n_heads, -1, seq_len)
        attention_mask = (1.0 - attention_mask) * -10000.0
        head_mask = [None] * self.n_layers # not used 

        # Attention
        q = self.query(X)[:, -1:, :].permute(1, 0, 2) # last query only
        k = self.key(X).permute(1, 0, 2)
        v = self.value(X).permute(1, 0, 2)
        # out, _ = self.attn(q, k, v, attn_mask=attention_mask)
        out, _ = self.attn(q, k, v)
        
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
        preds = out.view(batch_size, -1) 
        # preds = self.activation(out).view(batch_size, -1) 

        return preds
    

class LGCNGRUATTN(LGCNModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags,
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.gru = nn.GRU(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

    ## add
    def forward(self, test, question, tag, correct, mask, interaction, elapsed):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        elapsed=elapsed,
                                        )
        out, _ = self.gru(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        # print(f"mask.shape:{mask.shape}") # [64, 20]
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # print(f"extended_attention_mask.shape:{extended_attention_mask.shape}") # [64, 1, 1, 20]
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out
    
class LGCNLastQuery(LGCNModelBase):
    """
    embedding --> Multihead Attention --> LSTM
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 1,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 100,
        **kwargs    
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags,
        )
        self.n_heads = n_heads
        self.drop_out = drop_out

        # === Embedding (ModelBase)
        # hd: hidden dimension
        # intd: intermediate hidden dimension
        # hd, intd = hidden_dim, hidden_dim // 3

        # embedding_interaction (3 -> intd)
        # embedding_test        (n_tests + 1 -> intd)
        # embedding_question    (n_questions + 1 -> intd)
        # embedding_tag         (n_tags + 1 -> intd)
        # comb_proj             (intd * (len(features) + 1) -> hd) # Concatenated Embedding Projection

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

        # === Fully Connected Layer 
        # fc (hd -> 1)

        self.activation = nn.Sigmoid()


    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to("cuda")
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to("cuda")
        return (h, c)

    def forward(self, test, question, tag, correct, mask, interaction, elapsed):
        # print(f'interaction size: {interaction.size()}')
        seq_len = interaction.size(1)
        # print(f'seq_len : {seq_len}')
        # print(f'mask: {mask.size()}')

        # Embedding
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        elapsed=elapsed,
                                        )
        
        # TODO: Add new features
        # 일단은 ModelBase 에서 처리하는 방향으로 진행 예정
        # embed_timebin = self.embedding_timebin(timebin.int().to('cuda'))
        # X = torch.cat([X, embed_timebin.unsqueeze(1)], dim=2)
        
        # Encoder 
        # make attention_mask (size: batch_size * n_head, seq_len)
        attention_mask = mask.repeat(1, self.n_heads)
        attention_mask = attention_mask.view(batch_size * self.n_heads, -1, seq_len)
        attention_mask = (1.0 - attention_mask) * -10000.0
        head_mask = [None] * self.n_layers # not used 

        # Attention
        q = self.query(X)[:, -1:, :].permute(1, 0, 2) # last query only
        k = self.key(X).permute(1, 0, 2)
        v = self.value(X).permute(1, 0, 2)
        # out, _ = self.attn(q, k, v, attn_mask=attention_mask)
        out, _ = self.attn(q, k, v)
        
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
        preds = out.view(batch_size, -1) 
        # preds = self.activation(out).view(batch_size, -1) 

        return preds