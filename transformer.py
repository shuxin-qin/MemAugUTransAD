import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale
        if attn_mask is not None:
        	# 给需要mask的地方设置一个负无穷
        	attention = attention.masked_fill_(attn_mask, -np.inf)
		# 计算softmax
        attention = self.softmax(attention)
		# 添加dropout
        attention = self.dropout(attention)
		# 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=3, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


def ScaledDotProductAtt(q, k, v, scale):
    #print('q:',q.shape, 'k:',k.shape, 'v:', v.shape)
    scores = torch.bmm(q, k.permute(0, 2, 1)) / scale
    #print('scores:', scores.shape)
    attn = torch.softmax(scores, 2)
    res = torch.bmm(attn, v)
    #print('res:', res.shape)
    return res

class MultiHeadAttention1(nn.Module):

    def __init__(self, n_feature, n_head, dropout = 0.1):
        """Multihead Attention Module
        MultiHead(Q, K, V) = Concat(head_1, ..., head_n) W^o
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
        args:
            n_feature: the number of feature
            num_head: the number of heads
            dropout: the rate of dropout
        """
        super(MultiHeadAttention1, self).__init__()
        self.n_feature = n_feature
        self.n_head = n_head
        dk = n_feature // n_head
        self.scale = math.sqrt(dk)

        self.qfc = nn.Linear(n_feature, n_feature)
        self.kfc = nn.Linear(n_feature, n_feature)
        self.vfc = nn.Linear(n_feature, n_feature)
        self.ofc = nn.Linear(n_feature, n_feature)

        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(n_feature) 

    def forward(self, key, value, query, attn_mask=None):
        """
        shape:
            query,key,value: T x batch_size x n_feature
        """
        residual = query
        querys = self.qfc(query).chunk(self.n_head, -1)
        keys = self.kfc(key).chunk(self.n_head, -1)
        values = self.vfc(value).chunk(self.n_head, -1)
        
        context = torch.cat([ScaledDotProductAtt(q, k, v, self.scale) for q,k,v in zip(querys, keys, values)], -1)

        # final linear projection
        output = self.ofc(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output


class PositionalEncoding1(nn.Module):
    
    def __init__(self, d_model, max_seq_len):
        """初始化。
        
        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
          [pos / np.pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))
        
        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        
        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
          [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)
    

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 用了个技巧先计算log的在计算exp
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # position * div_term 这里生成一个以pos为行坐标，i为列坐标的矩阵
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)  # x.size(1)就是有多少个pos
        return self.dropout(x)


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim, ffn_dim, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.permute(0, 2, 1)
        output = self.w2(F.relu(self.w1(output)))
        output = output.permute(0, 2, 1)
        output = self.dropout(output)
        # add residual and norm layer
        # print('output:', output.shape)
        output = self.layer_norm(x + output)
        return output


def padding_mask(seq_k, seq_q):
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask


class EncoderLayer(nn.Module):
    """Encoder的一层"""

    def __init__(self, n_feature, num_heads, hid_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()

        #self.freqatt = FrequencyAttention(K = 4, dropout = dropout)
        self.attention = MultiHeadAttention(n_feature, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(n_feature, hid_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        # print('context:', context.shape)
        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class DecoderLayer(nn.Module):

    def __init__(self, n_feature, num_heads, hid_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(n_feature, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(n_feature, hid_dim, dropout)

    def forward(self,
              dec_inputs,
              enc_outputs,
              self_attn_mask=None,
              context_attn_mask=None):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(
          dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(
          enc_outputs, enc_outputs, dec_output, context_attn_mask)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention
