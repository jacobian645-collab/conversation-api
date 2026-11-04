# AI/ML tutor: Amazon Nova fine-tuning dataset

Ready-to-use training data for fine-tuning Amazon Nova (Micro / Lite / Pro) on Bedrock, in the
`bedrock-conversation-2024` JSONL format.

| File | Samples | Multi-turn |
|---|---|---|
| `train.jsonl` | 101 | 18 |
| `validation.jsonl` | 11 | 2 |

- **Persona (system prompt):** a concise AI/ML tutor for engineers and product managers: accurate, plain
  language, one concrete example, under 150 words.
- **Coverage:** ML fundamentals, evaluation metrics, classic models, deep learning, transformers and LLMs,
  prompting, RAG, fine-tuning (SFT, LoRA, RLHF, DPO, distillation), LLM evaluation and guardrails, agents and
  tool use, Amazon Bedrock, MLOps, and GenAI product metrics.
- **Checks:** passes `convo validate-jsonl` (Nova's role order, sample counts, length limit, reserved
  keywords); no duplicate questions; the split is stratified so validation includes multi-turn samples.
- **Origin:** written with Claude (Anthropic) for this portfolio project. If you use it beyond a demo,
  check the terms that apply to training a model on another provider's output.

```bash
convo validate-jsonl examples/datasets/ai-ml-tutor/train.jsonl --validation examples/datasets/ai-ml-tutor/validation.jsonl
bash scripts/bedrock_finetune_nova.sh examples/datasets/ai-ml-tutor/train.jsonl examples/datasets/ai-ml-tutor/validation.jsonl
```
