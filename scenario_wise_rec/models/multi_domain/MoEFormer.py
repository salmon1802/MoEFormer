import torch
import torch.nn as nn
import torch.nn.functional as F
from ...basic.layers import MLP, EmbeddingLayer

class MoEFormer(nn.Module):
    def __init__(self, features, domain_num, hidden_dim, num_layers, 
                 num_semantic_tokens=32, num_domain_tokens=1, expand_ratio=2):
        super().__init__()
        self.features = features
        self.embedding = EmbeddingLayer(features)
        self.embedding_dim = features[0].embed_dim
        self.num_feature = len(features)
        self.domain_num = domain_num
        self.num_layers = num_layers
        
        # --- Token 数量定义 ---
        self.num_field_tokens = self.num_feature
        self.num_semantic_tokens = num_semantic_tokens
        self.num_domain_tokens = num_domain_tokens # 每个域的Token数 (k) 即 domain token + reasoning token
        self.num_total_domain_reasoning_tokens = domain_num * num_domain_tokens
        
        # 总 Token 数 = f + m + (n * k) 这里直接将domain token与reasoning token合并以简化操作
        self.total_num_tokens = self.num_field_tokens + self.num_semantic_tokens + self.num_total_domain_reasoning_tokens
        
        # --- 维度定义 ---
        # token_dim: 每个 Token 的维度
        token_dim = hidden_dim * self.num_feature
        self.token_dim = token_dim
        
        # semantic Tokens 部分的展开维度
        self.semantic_hidden_dim = hidden_dim * (self.num_semantic_tokens + self.num_total_domain_reasoning_tokens)
        
        # --- 模型层定义 ---
        self.hhtu_layers = nn.ModuleList([
            HHTU(input_dim=token_dim,
                attention_dim=token_dim * 2,
                expand_ratio=expand_ratio,
                num_tokens=self.total_num_tokens 
                )
            for _ in range(num_layers)
        ])
        
        self.predictor = nn.ModuleList(
            nn.Sequential(nn.Linear(token_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)) for i in range(self.domain_num)
        )

        self.pre_norm = nn.ModuleList([
            nn.LayerNorm(token_dim)
            for _ in range(num_layers)
        ])
        
        # --- 投影参数 ---
        self.emb2semantic = nn.Parameter(torch.empty(self.num_feature, self.embedding_dim, self.semantic_hidden_dim))
        self.emb2field = nn.Parameter(torch.empty(self.num_feature, self.embedding_dim, self.token_dim))
        
        self.reset_parameters()

        # --- Mask 构建 ---
        self._build_mask()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb2semantic)
        nn.init.xavier_uniform_(self.emb2field)

    def _build_mask(self):
        """
        [Modified] 构建 V17 Mask 逻辑: 域内使用下三角 (Causal) Mask
        
        逻辑:
        1. Public Block (Field + semantic): 互相可见 (Full Attention)。
        2. Domain Block: 
           - 可见 Public Block。
           - 域内可见性: 下三角自回归 Mask。
           - 域间不可见。
        """
        # 初始化为负无穷 (表示不可见)
        mask = torch.full((self.total_num_tokens, self.total_num_tokens), -1e9)
        
        # Public Tokens 数量 (Field + semantic)
        num_public = self.num_field_tokens + self.num_semantic_tokens
        
        # 1. Public -> Public (Field & semantic 互看)
        mask[:num_public, :num_public] = 0.0
        
        # 2. Domain -> Public (所有 Domain Token 都能看 Public)
        mask[num_public:, :num_public] = 0.0
        
        # 3. Domain -> Domain (处理域间不可见，域内下三角)
        k = self.num_domain_tokens
        
        # 创建一个 k*k 的下三角矩阵，下三角为0(可见)，上三角为-inf(不可见)
        # torch.tril(torch.ones(...)) 生成下三角为1的矩阵
        causal_mask = torch.tril(torch.ones(k, k))
        # 将 1 映射为 0.0 (可见), 0 映射为 -1e9 (不可见)
        local_mask_val = torch.zeros(k, k).masked_fill(causal_mask == 0, -1e9)
        
        # 遍历每个域，填入下三角 Mask
        for d in range(self.domain_num):
            start_idx = num_public + d * k
            end_idx = start_idx + k
            
            # [Modified] 域内使用下三角 Mask
            mask[start_idx:end_idx, start_idx:end_idx] = local_mask_val
        
        self.register_buffer('attn_mask', mask, persistent=False)

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()
        
        # tokens shape: batch_size × total_num_tokens × token_dim
        tokens = self.tokenizer(x) 
        
        mask = self.attn_mask
        
        for i in range(self.num_layers):
            tokens = self.hhtu_layers[i](self.pre_norm[i](tokens), mask=mask) + tokens
            
        # --- 输出聚合 ---
        ys = []
        
        # Domain Tokens 起始位置
        base_idx = self.num_field_tokens + self.num_semantic_tokens
        
        for d in range(self.domain_num):
            # 获取当前域 d 对应的所有 tokens
            d_start = base_idx + d * self.num_domain_tokens
            d_end = d_start + self.num_domain_tokens
            
            # Shape: [Batch, k, token_dim]
            domain_tokens_group = tokens[:, d_start:d_end, :]
            
            # [Modified] 聚合策略: 取最后一个 Token (Last Token Prediction)
            # 最后一个 Token 能够看到该域内之前所有的 Token (由于下三角Mask)
            # Shape: [Batch, token_dim]
            domain_repr = domain_tokens_group[:, -1, :] 
            
            # 预测
            y_pred = self.predictor[d](domain_repr)
            ys.append(y_pred)
        
        ys_stack = torch.stack(ys, dim=1)
        ys_stack = torch.sigmoid(ys_stack)
        
        gather_index = domain_id.long().view(-1, 1, 1)
        final = ys_stack.gather(1, gather_index).squeeze()
        return final

    def tokenizer(self, x):
        # 1. 获取原始特征 Embedding: [B, F, D]
        feature_embs = self.embedding(x, self.features, squeeze_dim=False)  
        # --- 分支 A: 生成 Field Tokens [B, T_field, TokenDim] ---
        field_tokens = torch.einsum("BFD,FDK->BFK", feature_embs, self.emb2field)
 
        # --- 分支 B: 生成 semantic Tokens (semantic + All Domain Tokens) ---
        # Einsum: [B, F, D] x [F, D, T_semantic * Hidden_Dim] -> [B, F, T_semantic * Hidden_Dim]
        semantic_raw = torch.einsum("BFD,FDH->BFH", feature_embs, self.emb2semantic)
        B, F, _ = semantic_raw.shape
        T_semantic = self.num_semantic_tokens + self.num_total_domain_reasoning_tokens
 
        # View: [B, F, T_semantic, Hidden_Dim] 
        semantic_tokens = semantic_raw.view(B, F, T_semantic, -1)
 
        # Permute: [B, T_semantic, F, Hidden_Dim]
        semantic_tokens = semantic_tokens.permute(0, 2, 1, 3)
 
        # Flatten: [B, T_semantic, TokenDim]
        semantic_tokens = semantic_tokens.reshape(B, T_semantic, -1)
 
        # Concat: [B, T_all, TokenDim]
        all_tokens = torch.cat([field_tokens, semantic_tokens], dim=1)
        return all_tokens

class HHTU(nn.Module):
    def __init__(self,
                 input_dim=64,
                 attention_dim=64,
                 expand_ratio=2,
                 use_scale=True,
                 num_tokens=None): 
        super(HHTU, self).__init__()
        self.scale_factor = (attention_dim ** -0.5) if use_scale else 1.0
        self.num_tokens = num_tokens
        self.W_q = nn.Parameter(torch.empty(num_tokens, input_dim, attention_dim))
        self.W_k = nn.Parameter(torch.empty(num_tokens, input_dim, attention_dim))
        self.W_v = nn.Parameter(torch.empty(num_tokens, expand_ratio, input_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, x, mask=None):
        query = torch.einsum('bti,tid->btd', x, self.W_q)
        key = torch.einsum('bti,tid->btd', x, self.W_k)
        value = torch.einsum("bti,teik->btk", x, self.W_v) * x        
        output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask, 
            scale=self.scale_factor
        )
        return output