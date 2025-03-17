# LJMU-MS-Research-GenAI
*************LJMU MS Research in GenAI**********

PARAPHRASE IDENTIFICATION AND COMPARATIVE ANALYSIS WITH FINE TUNED AND PROMPT BASED LLM MODEL

Paraphrase Identification is a critical task in Natural Language Processing (NLP) that 
determines whether two sentences convey the same meaning. This research investigates the 
effectiveness of prompt-based, fine-tuned encoder-based, decoder-based, and encoder-decoder
based Large Language Models (LLMs) for paraphrase identification. Using the Microsoft 
Research Paraphrase Corpus (MRPC), we evaluate models across four approaches: prompt
based (GPT-4o-mini, GPT-3.5-Turbo, Llama-3.2-1B, Llama-3.2-3B, Mistral-7B), fine-tuned 
decoder-based (Llama-3.2-1B, Llama-3.2-3B, Mistral-7B), fine-tuned encoder-based 
(ModernBERT), and fine-tuned encoder-decoder-based (T5). The study employs fine-tuning 
strategies such as QLoRA (Quantized Low-Rank Adaptation) & LoRA (Low-Rank Adaptation) 
to optimize model performance while minimizing computational costs. Results show that fine
tuned models consistently outperform prompt-based approaches in terms of accuracy. Among 
fine-tuned models, encoder-based models achieve the highest accuracy (88%), followed closely 
by encoder-decoder-based models (86%) and decoder-based models (84%). While decoder
based models perform well, they demand significantly higher computational resources, making 
them less efficient for large-scale applications. In contrast, encoder-based models achieve 
comparable accuracy with reduced computational overhead, making them a more practical 
alternative. Prompt-based models, while computationally efficient, achieve the lowest accuracy 
(76%) and are highly sensitive to prompt design, leading to variability in responses. This study 
contributes to NLP by providing insights into the trade-offs between these approaches and 
recommending efficient fine-tuning techniques for resource optimization. These findings guide 
model selection based on accuracy, computational constraints, and deployment feasibility in 
real-world applications.
