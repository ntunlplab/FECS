# Fidelity-Enriched Contrastive Search: Reconciling the Faithfulness-Diversity Trade-Off in Text Generation

This is the official repository of our paper [Fidelity-Enriched Contrastive Search: Reconciling the Faithfulness-Diversity Trade-Off in Text Generation](https://arxiv.org/pdf/2310.14981.pdf), *EMNLP* 2023.

*TL;DR*: This work proposes FECS, a novel decoding method with context-aware regularization terms to mitigate hallucination while preserving generation diversity.

---

### Try FECS decoding with Huggingface 🤗
Steps:
1. Install ```transformers``` version ```4.24.0```
   
2. Integrate FECS into the transformers package by the following command
   
    ```
    cp src/generation_utils.py [your_environment_path]/python3.X/site-packages/transformers/generation_utils.py
    ```
    
3. Try FECS with the auto-regressive LMs of your choice. Following is an example on the abstractive summarization task.
   
    ```python
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_name == "facebook/opt-6.7b"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    model.eval()
    
    input = "Article:[shot_1_article]\nSummary:[shot_1_summary]\n...\nArticle:[shot_n_article]\nSummay:[shot_n_summary]\nArticle:[test_input_article]"
    input_ids = tokenizer(input, return_tensors='pt').input_ids 
    _, prefix_len = input_ids.size()
    input_ids = input_ids.to(device)

    source = input.split('Article:')[-1] # The knowledge source to be faithful to (e.g., for abstractive summarization, the source is the article to be summarized).
    source_ids = tokenizer(source, return_tensors='pt').input_ids 
    _, source_len = source_ids.size()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            top_k=4, # The size of the candidate token set. The candidate tokens are the top-k probability tokens from the model’s prediction distribution.
            penalty_alpha=0.3, # The weight for the degeneration penalty. A larger alpha promotes candicate tokens which are more diverse.
            source_penalty_beta=0.3, # The weight for the faithfulness reward. A larger beta promotes candidate tokens which are more similar to the source.
            block_context=True, # Set this to "True" so that degeneration penalty is only applied on the generated content, instead of the given input content (i.e., the prefix).
            prefix_len=prefix_len,
            source_len=source_len,
            max_length=prefix_len+512
        )
    
    output = tokenizer.decode(output_ids[0][prefix_len:], skip_special_tokens=True)
    ```
   💡 Try different ```top_k```, ```penalty_alpha```, ```source_penalty_beta``` values to get the desired output for your tasks.
<br>

<details><summary>Reproduce paper experiments</summary>
  
*WIP*

</details>
