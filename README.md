## LLM Knowledge Evaluations

### Overview of results

| Model       | Baseline Score | Vector RAG |Vector RAG (with re-ranking) | Vector RAG (with re-ranking on basic chunking|
|----------   |----------      |----------  |----------| ----------|
|granite3.3:2b|(45.83%)|(49.31%) |(45.83%)| (43.75%) |
|qwen3:1.7b   |(59.72%)|(56.94%) |(64.58%)| (58.33%) |
|qwen3:4b     |(72.92%)|(75.00%) |(77.78%)| (77.08%) |
|gemma3:1b |(31.25%)|(20.83%) |(23.61%)| (21.53%) |
|gemma3:4b |(58.33%)|(60.42%) |(61.11%)| (64.58%) |
|phi4-mini:3.8b |(68.75%)|(65.97%) |(72.22%)| (70.14%) |

### Result Details

#### Basic

The model is prompted with the question and expected to return the answer without specifying any context. In this approach the model is forced to rely upon its parametric knowledge.

- Model Input: Question
- Expected Output: Answer

| Model | Correct Answers | Pass % |
|----------|----------|----------|
|granite3.3:2b | 66/144 |(45.83%)|
|qwen3:1.7b | 86/144 |(59.72%)|
|qwen3:4b | 105/144 | (72.92%)|
|gemma3:1b | 45/144 |(31.25%)|
|gemma3:4b | 84/144 |(58.33%)|
|phi4-mini:3.8b | 99/144 |(68.75%)|

#### Vector RAG on title based chunking

- Model Input: Question + Top 3 results retrieved from vector search
- Expected Output: Answer

| Model | Correct Answers | Pass % |
|----------|----------|----------|
|granite3.3:2b | 71/144 | (49.31%)|
|qwen3:1.7b | 82/144 | (56.94%)|
|qwen3:4b | 108/144 | (75.00%)|
|gemma3:1b | 30/144 | (20.83%)|
|gemma3:4b | 87/144 | (60.42%)|
|phi4-mini:3.8b | 95/144 | (65.97%)|

#### Vector RAG: with result reranking on title based chunking

- Model Input: Question + Top 3 results retrieved from vector search which are reranked using a cross-encoder.
- Expected Output: Answer

| Model | Correct Answers | Pass % |
|----------|----------|----------|
|granite3.3:2b | 66/144 | (45.83%)|
|qwen3:1.7b | 93/144 | (64.58%)|
|qwen3:4b | 112/144 | (77.78%)|
|gemma3:1b | 34/144 | (23.61%)|
|phi4-mini:3.8b | 104/144 | (72.22%)|
|gemma3:4b | 88/144 | (61.11%)|

#### Vector RAG: with result reranking on basic chunking with overlaps

- Model Input: Question + Top 3 results retrieved from vector search which are reranked using a cross-encoder.
- Expected Output: Answer

| Model | Correct Answers | Pass % |
|----------|----------|----------|
|granite3.3:2b | 63/144 | (43.75%)|
|qwen3:1.7b | 84/144 | (58.33%)|
|qwen3:4b | 111/144|  (77.08%)|
|gemma3:1b | 31/144 | (21.53%)|
|phi4-mini:3.8b | 101/144|  (70.14%)|
|gemma3:4b | 93/144 | (64.58%)|
