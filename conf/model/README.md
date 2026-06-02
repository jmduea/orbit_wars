# Model configs

All policy/value model architecture knobs belong here.

Typical fields:

```yaml
architecture: planet_graph_transformer
pointer_decoder: factorized_topk
hidden_size: 128
attention_heads: 4
max_moves_k: 2
value_head: distributional
```

Examples:

```bash
uv run ow train model=transformer_factorized
uv run ow train model=transformer_factorized_small
```
