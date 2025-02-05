# Databricks notebook source
# MAGIC %md
# MAGIC # Generate Synthetic Evaluation Datasets for Retriever Evaluation
# MAGIC
# MAGIC Leverage previously chunked datasets to generate questions, answers and additional metadata to be used specfically for vector database **retriever** evaluation

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade langchain langchain-openai openai pandas seaborn scikit-learn mlflow python-dotenv
# MAGIC %restart_python

# COMMAND ----------

from importlib.metadata import version

print(f"langchain: {version('langchain')}")
print(f"langchain-openai: {version('langchain-openai')}")
print(f"openai: {version('openai')}")
print(f"pandas: {version('pandas')}")
print(f"seaborn: {version('seaborn')}")
print(f"sklearn: {version('scikit-learn')}")
print(f"mlflow: {version('mlflow')}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Load envrionment variables from **.env** files. If **OPENAI_API_KEY** is provided then **OpenAI** will be the default provider, **Databricks** will be used otherwise.

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from typing import Any, Dict, List

from mlflow.models import ModelConfig


model_config: ModelConfig = ModelConfig(development_config="model_config.yaml")

chunk_source: Dict[str, Any] = model_config.get("chunk_source")
chunk_schema: Dict[str, str] = chunk_source.get("schema")

chunk_source_table_name: str = chunk_source.get("table_name")
chunk_id_column: str = chunk_schema.get("chunk_id_column")
chunk_content_column: str = chunk_schema.get("chunk_content_column")
chunk_source_column: str = chunk_schema.get("chunk_source_column")

generation: Dict[str, Any] = model_config.get("generation")
num_examples: int = int(generation.get("num_examples"))
generated_evaluation_table_name: str = generation.get("table_name")

models: Dict[str, str] = model_config.get("models")
generation_model: str = models.get("generation_model")

print(f"chunk_source_table_name: {chunk_source_table_name}")
print(f"chunk_id_column: {chunk_id_column}")
print(f"chunk_content_column: {chunk_content_column}")
print(f"chunk_source_column: {chunk_source_column}")
print(f"generated_evaluation_table_name: {generated_evaluation_table_name}")
print(f"num_examples: {num_examples}")
print(f"generation_model: {generation_model}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Leverage caching for LLM prompt/responses as well as embeddings.

# COMMAND ----------

import json
import os
from os import PathLike

# For cost-saving, create a cache for the LLM responses
import threading

# For data analysis and visualization
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd

# For scraping
import requests
import seaborn as sns
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Cache:
    def __init__(self, persist_path: PathLike, cache_loading_fn):
        """
        The cache_loading_fn should be a function that takes arbitrary
        serializable arguments and returns a serilaizable value.
          value = cache_loading_fn(**kwargs)
        For example, for openai.chat.completions.create(...), the
        cache_loading_fn should be:
          def cache_loading_fn(**kwargs):
            result = openai.chat.completions.create(**kwargs)
            return result.to_dict_recursive()
        """
        if isinstance(persist_path, Path):
            persist_path = persist_path.as_posix()
            
        self._cache = self._get_or_create_cache_dict(persist_path)
        self._persist_path = persist_path
        self._cache_loading_fn = cache_loading_fn
        self._cache_lock = threading.Lock()

    @classmethod
    def _get_or_create_cache_dict(cls, persist_path):
        if os.path.exists(persist_path):
            # File exists, load it as a JSON string into a dict
            with open(persist_path) as f:
                cache = json.load(f)
        else:
            # File does not exist, create an empty dict
            cache = {}
        return cache

    def _save_to_file(self):
        with open(self._persist_path, "w") as file:
            json.dump(self._cache, file)

    def _update_cache(self, key, value):
        with self._cache_lock:
            self._cache[key] = value
            self._save_to_file()

    def get_from_cache_or_load_cache(self, **kwargs):
        key = json.dumps(kwargs)

        with self._cache_lock:
            value = self._cache.get(key, None)

        if value is None:
            value = self._cache_loading_fn(**kwargs)
            self._update_cache(key, value)
        else:
            print("Loaded from cache")

        return value


def chat_completion_create_fn(**kwargs):
    result = openai.chat.completions.create(**kwargs)
    return result.to_dict()


def cached_openai_chat_completion_create(**kwargs):
    cache = kwargs.pop("cache")
    return cache.get_from_cache_or_load_cache(**kwargs)


def embeddings_embed_documents_fn(**kwargs):
    chunk = kwargs.get("chunk")
    return embeddings.embed_documents([chunk])


def cached_langchain_openai_embeddings(**kwargs):
    cache = kwargs.pop("cache")
    return cache.get_from_cache_or_load_cache(**kwargs)


# COMMAND ----------

from typing import Any, Dict
from pathlib import Path

# Other configurations

# Choose a seed for reproducible results
seed: int = 123

# For cost-saving purposes, choose a path to persist the responses for LLM calls
cache_path: Path = "_cache.json"
embeddings_cache_path: Path = "_embeddings_cache.json"



# COMMAND ----------

cache: Cache = Cache(cache_path, chat_completion_create_fn)
embeddings_cache: Cache = Cache(embeddings_cache_path, embeddings_embed_documents_fn)


# COMMAND ----------

def get_raw_response(content):
    prompt = f"""Please generate a question asking for the key information in the given paragraph.
    Also answer the questions using the information in the given paragraph.
    Please ask the specific question instead of the general question, like
    'What is the key information in the given paragraph?'.
    Please generate the answer using as much information as possible.
    If you are unable to answer it, please generate the answer as 'I don't know.'
    The answer should be informative and should be more than 3 sentences.

    Paragraph: {content}

    Please call the submit_function function to submit the generated question and answer.
    """

    messages = [{"role": "user", "content": prompt}]

    submit_function = {
        "name": "submit_function",
        "description": "Call this function to submit the generated question and answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question asking for the key information in the given paragraph.",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer to the question using the information in the given paragraph.",
                },
            },
            "required": ["question", "answer"],
        },
    }

    return cached_openai_chat_completion_create(
        messages=messages,
        model=generation_model,
        functions=[submit_function],
        function_call="auto",
        temperature=0.0,
        seed=seed,
        cache=cache,
    )


def generate_question_answer(content):
    if content is None or len(content) == 0:
        return "", "N/A"

    response = get_raw_response(content)
    try:
        func_args = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
        question = func_args["question"]
        answer = func_args["answer"]
        return question, answer
    except Exception as e:
        return str(e), "N/A"


# COMMAND ----------

from pyspark.sql import DataFrame
import random

def sample_rows(df: DataFrame, n: int, seed: int = None) -> DataFrame:
    """
    Sample n rows from the DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame to sample from.
    n (int): The number of rows to sample.
    seed (int, optional): The seed for the random number generator.
    
    Returns:
    DataFrame: A DataFrame with n sampled rows.
    """
    fraction: float = n / df.count()
    sampled_df: DataFrame = df.sample(withReplacement=False, fraction=fraction, seed=seed)
    return sampled_df.limit(n)
  

# COMMAND ----------

from pyspark.sql import DataFrame
import pandas as pd


chunks_df: DataFrame = spark.table(chunk_source_table_name)

sampled_df: DataFrame = sample_rows(chunks_df, num_examples, seed=seed)

filtered_pdf: pd.DataFrame = sampled_df.toPandas()

display(filtered_pdf)

# COMMAND ----------


get_raw_response(filtered_pdf[chunk_content_column][0])



# COMMAND ----------

from typing import List, Dict

queries: List[Dict[str, str]] = []

n: int = len(filtered_pdf)
for i, row in filtered_pdf.iterrows():
    chunk: str = row[chunk_content_column]
    question: str
    answer: str
    question, answer = generate_question_answer(chunk)
    print(f"{i+1}/{n}: {question}")
    queries.append(
        {
            "question": question,
            "answer": answer,
            "chunk": chunk,
            "chunk_id": row[chunk_id_column],
            "source": row[chunk_source_column],
        }
    )


# COMMAND ----------

result_pdf = pd.DataFrame(queries)
result_pdf = result_pdf[result_pdf["answer"] != "N/A"]
display(result_pdf)

# COMMAND ----------

result_df: DataFrame = spark.createDataFrame(result_pdf)
result_df.write.format("delta").mode("overwrite").saveAsTable(generated_evaluation_table_name)

# COMMAND ----------

all_result_pdf: pd.DataFrame = spark.table(generated_evaluation_table_name).toPandas()
all_result_pdf.head(3)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Evaluating Diversity of Questions
# MAGIC Diversity of questions is important because we want questions to cover the majority of the document content. In addition, we want to be able to evaluate the retriever with different forms of questioning. We want to be able to have harder questions and easier questions. All of these are not straightforward to analyze, and we decided to analyze its through question length and latent space embeddings.
# MAGIC
# MAGIC ### Length
# MAGIC Length gives a sense of how diverse the questions are. Some questions may be wordy while others are straight to the point. It also allows us to identify problems with the question generated.

# COMMAND ----------

# Length
questions = all_result_pdf["question"].to_list()
question_len = pd.DataFrame([len(q) for q in questions], columns=["length"])
question_len.hist(bins=5)
plt.title("Histogram of Question Lengths")
plt.xlabel("Question Length")
plt.ylabel("Frequency")
plt.show()


# COMMAND ----------

# Calculating percentile values
p10 = int(question_len["length"].quantile(0.10))
p90 = int(question_len["length"].quantile(0.90))

print("p10-p90 range is", p90 - p10)

#80% of the text lengths vary by at most 35 characters/words/tokens


# COMMAND ----------

[q for q in questions if len(q) > 100]


# COMMAND ----------

benchmark_questions = [
    "What is MLflow?",
    "What is MLflow about?",
    "What is MLflow Tracking?",
    "What is MLflow Evaluation?",
    "Why is RAG so popular?",
]
questions_to_embed = questions + benchmark_questions


# COMMAND ----------

# Apply embeddings
embeddings = OpenAIEmbeddings()
question_embeddings = embeddings.embed_documents(questions_to_embed)
# PCA on embeddings to reduce to 10-dim
pca = PCA(n_components=10)
question_embeddings_reduced = pca.fit_transform(question_embeddings)
# TSNE on embeddings to reduce to 2-dim
perplexity = min(30, question_embeddings_reduced.shape[0] - 1)
tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
lower_dim_embeddings = tsne.fit_transform(question_embeddings_reduced)


# COMMAND ----------

labels = np.concatenate(
    [
        np.full(len(lower_dim_embeddings) - len(benchmark_questions), "generated"),
        np.full(len(benchmark_questions), "benchmark"),
    ]
)
data = pd.DataFrame(
    {"x": lower_dim_embeddings[:, 0], "y": lower_dim_embeddings[:, 1], "label": labels}
)
sns.scatterplot(data=data, x="x", y="y", hue="label")


# COMMAND ----------

all_result_pdf.sample(3)


# COMMAND ----------

embedded_queries = all_result_pdf.copy()
embedded_queries["chunk_emb"] = all_result_pdf["chunk"].apply(
    lambda x: np.squeeze(cached_langchain_openai_embeddings(chunk=x, cache=embeddings_cache))
)
embedded_queries["question_emb"] = all_result_pdf["question"].apply(
    lambda x: np.squeeze(cached_langchain_openai_embeddings(chunk=x, cache=embeddings_cache))
)


# COMMAND ----------

def cossim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


embedded_queries["cossim"] = embedded_queries.apply(
    lambda row: cossim(row["question_emb"], row["chunk_emb"]), axis=1
)


# COMMAND ----------

scores = embedded_queries["cossim"].to_list()
plt.hist(scores, bins=5)


# COMMAND ----------

mask = embedded_queries["cossim"] < 0.75
lower_cossim = embedded_queries[mask]
for i, row in lower_cossim.iterrows():
    print(f"Question: {i}")
    print(row["question"])
    print("Chunk:")
    print(row["chunk"])
    print("cossim:")
    print(row["cossim"])
