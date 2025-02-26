# vector-db-bakeoff

**Databricks** has developed tools to evaluate how well Generative AI applications perform, including their ability to retrieve relevant information. While our existing tools assess overall AI application quality, this project focuses specifically on **vector database** performance, measuring how efficiently different databases retrieve information.

This utility generates **synthetic** test questions and answers to benchmark retrieval performance across various vector databases. It helps fine-tune parameters like **chunk size** and **overlap** while providing **side-by-side** comparisons of different database implementations.

Additionally, it allows users to configure and track key performance metrics such as latency, precision, and recall at different levels (e.g., **precision_at_k(4)**, **recall_at_k(5)**). It also supports custom evaluation metrics, ensuring flexibility in assessing retrieval quality. Built on **MLflow** evaluation, it provides easy tracking and comparison, helping teams identify the best performing vector database for AI-driven search and retrieval use cases.


## Use Cases

This tool can be used to:

* Compare vector database performance to determine the best option for AI-driven retrieval.
* Optimize retrieval strategies by tuning hyperparameters like chunk size, overlap, and indexing methods.
* Measure precision, recall, and latency to understand how well a database retrieves relevant results.
* Validate the impact of different vector storage solutions on overall application performance.
* Develop and test custom evaluation metrics to align with specific business needs.

## Prerequisites

The framework assumes that each target Vector Store has been populated individually using identical hyperparameters (ie chunk size, overlap, strategy)  
Vector Store initialization parameters must be configured in ```model_config.yaml```

## Requirements
- langchain
- langchain-openai
- databricks-langchain
- lanchain-chroma
- langchain-milvus
- pandas
- seaborn
- sklearn
- mlflow

## Synthetic Data Generation
```01_question-generation.py```

### Usage

Synthetic data only needs to be generated once and can then be shared across evaluation runs

1. **Load Environment Variables**: Ensure that environment variables are loaded from `.env` files. If `OPENAI_API_KEY` is provided, OpenAI will be the default provider; otherwise, Databricks will be used.

2. **Configure Model and Chunk Source**: Define the model configuration and chunk source details in `model_config.yaml`.

3. **Leverage Caching**: Use caching for LLM prompt/responses and embeddings to save costs.

4. **Generate Questions and Answers**: Use the provided functions to generate questions and answers from the chunked datasets.

5. **Sample Rows**: Sample rows from the DataFrame for evaluation.

6. **Evaluate Diversity**: Evaluate the diversity of generated questions using question length and latent space embeddings.

## Retriever Evaluation
```02_retriever-evaluation.py```

### Usage

A convenience class ```VectorStoreFactory``` has been provided to instantiate instances of vector store retriever.  
Example configurations and factories have been provided for **Chroma**, **Milvius** and **Databricks**, however, additional vector stores can easily be added.

1. Automate the bakeoff execution by providing a set of preconfigured **competitors** to be evaluated.
2. The results can be viewed as mlflow experiment/run metrics

#### Default Metrics

The following metrics are included by default, however, any desired, supported built-in or custom metrics can be configured.
- mlflow.metrics.latency()
- mlflow.metrics.precision_at_k(4)
- mlflow.metrics.recall_at_k(4)
- mlflow.metrics.precision_at_k(5)
- mlflow.metrics.recall_at_k(5)

## Configuration

Evaluation and Vector Store instance configuration is stored in ```model_config.yaml```

### Example:

```python

# Configuration for synthetic data generation

generation:
  num_examples: 25                                          # The number of example to generate
  table_name: nfleming.united_airlines.generated_evaluation # The table to store the generated examples

chunk_source:
  table_name: nfleming.united_airlines.documents_chunked    # The table which contains the chunked documents
  schema:                                                   # The schema of this table
    chunk_id_column: id
    chunk_content_column: content
    chunk_source_column: source

# Identify LLMs to be used for difference roles
models:
  embedding_model: openai                                    # Embedding Model
  chat_model: openai                                         # Chat Model 
  judge_model: openai:/gpt-4                                 # LLM Judge
  generation_model: gpt-4o-mini                              # LLM Used for Generation


# Configure Vector Store instances 
vector_stores:
  databricks:                                               # Unique name / alias
    type: databricks                                        # The vector store type
    options:                                                # Options to be provied to the factory function for instantiation
      endpoint: one-env-shared-endpoint-12
      index_name: pandas_poc.paws.documents_chunked_vs_index
      columns: 
      - source

  milvus:                                                  # Unique name / alias
    type: milvus                                           # The vector store type
    options: ~                                             # Options to be provied to the factory function for instantiation

  chroma:
    type: chroma
    options: ~

# Define the parameters to be used for the bakeoff
bakeoff:
  search_kwargs:
    k: 5

# Provide the list of vector search instances (by their unique name) to be included for comparison
  competitors:
  - databricks
  - chroma
```

## License

This project is licensed under the MIT License.

