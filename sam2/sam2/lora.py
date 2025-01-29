from torch import nn, Tensor
import torch
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, init_std=1e-3):
        super(LoRALayer, self).__init__()
        self.low_rank_down = nn.Linear(in_features, rank, bias=False)
        self.low_rank_up = nn.Linear(rank, out_features, bias=False)

        # Инициализация весов LoRA для нулевого выхода
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std):
        # Инициализируем веса так, чтобы результат был равен нулю
        nn.init.normal_(self.low_rank_down.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.low_rank_up.weight, mean=0.0, std=init_std)

    def forward(self, x):
        return self.low_rank_up(self.low_rank_down(x))

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRALinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_adapter = LoRALayer(in_features, out_features, rank)

    def forward(self, x):
        # Сначала обычный линейный слой, затем LoRA адаптер
        return self.linear(x) + self.lora_adapter(x)

class MLPWithLoRA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
        rank: int = 32
    ) -> None:
        super(MLPWithLoRA, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        # Используем LoRALinearLayer вместо обычных nn.Linear
        self.layers = nn.ModuleList(
            LoRALinearLayer(n, k, rank) for n, k in zip([input_dim] + h, h + [output_dim])
        )

        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)

        if self.sigmoid_output:
            x = torch.sigmoid(x)

        return x

class AttentionWithLoRA(nn.Module):
    """
    An attention layer with optional LoRA adapters for the linear projections.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
        rank: int = 8,  # Rank for LoRA adapters
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        # Initialize linear layers with optional LoRA adapters
        self.q_proj = LoRALinearLayer(embedding_dim, self.internal_dim, rank)
        self.k_proj = LoRALinearLayer(self.kv_in_dim, self.internal_dim, rank)
        self.v_proj = LoRALinearLayer(self.kv_in_dim, self.internal_dim, rank)
        self.out_proj = LoRALinearLayer(self.internal_dim, embedding_dim, rank)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out