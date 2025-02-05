# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Evaluate Vector Stores

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade  langchain langchain-community langchain-openai openai mlflow python-dotenv
# MAGIC %pip install --quiet --upgrade langchain-chroma langchain-milvus databricks-langchain
# MAGIC %restart_python

# COMMAND ----------

from importlib.metadata import version

print(f"langchain: {version('langchain')}")
print(f"langchain-community: {version('langchain-community')}")
print(f"databricks-langchain: {version('databricks-langchain')}")
print(f"openai: {version('openai')}")
print(f"langchain-openai: {version('langchain-openai')}")
print(f"langchain-chroma: {version('langchain-chroma')}")
print(f"langchain-milvus: {version('langchain-milvus')}")
print(f"mlflow: {version('mlflow')}")

# COMMAND ----------

from dotenv import load_dotenv, find_dotenv

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
generated_evaluation_table_name: str = generation.get("table_name")

models: Dict[str, str] = model_config.get("models")
embedding_model: str = models.get("embedding_model")
chat_model: str = models.get("chat_model")
judge_model: str = models.get("judge_model")

bakeoff: Dict[str, Any] = model_config.get("bakeoff")
search_kwargs: Dict[str, Any] = bakeoff.get("search_kwargs")


print(f"generated_evaluation_table_name: {generated_evaluation_table_name}")
print(f"chunk_source_table_name: {chunk_source_table_name}")
print(f"chunk_id_column: {chunk_id_column}")
print(f"chunk_content_column: {chunk_content_column}")
print(f"embedding_model: {embedding_model}")
print(f"chat_model: {chat_model}")
print(f"judge_model: {judge_model}")
print(f"search_kwargs: {search_kwargs}")


# COMMAND ----------

context = dbutils.entry_point.getDbutils().notebook().getContext()

workspace_host: str = spark.conf.get("spark.databricks.workspaceUrl")
base_url: str = f"https://{workspace_host}/serving-endpoints/"
api_key: str = context.apiToken().get()

# COMMAND ----------

from typing import Any, Dict, List, Optional

from mlflow.models import ModelConfig 

from langchain.chains.base import Chain
from langchain_core.documents.base import Document
from langchain_core.document_loaders.base import BaseLoader
from langchain_community.document_loaders.pyspark_dataframe import PySparkDataFrameLoader
from langchain_core.vectorstores.base import VectorStore



from pyspark.sql import DataFrame


def databricks_vector_search(
  endpoint: str, 
  index_name: str, 
  columns: Optional[List[str]] = None) -> VectorStore:
  from databricks_langchain import DatabricksVectorSearch
  return DatabricksVectorSearch(endpoint=endpoint, index_name=index_name, columns=columns)
  
  
def chroma_vector_search() -> VectorStore:
  from langchain.vectorstores import Chroma
  chunk_table_df: DataFrame = spark.table(chunk_source_table_name)
  loader: BaseLoader = PySparkDataFrameLoader(spark, df=chunk_table_df, page_content_column=chunk_content_column)
  texts: List[Document] = loader.load()
  return Chroma.from_documents(texts, embeddings)


def milvus_vector_search() -> VectorStore:
  from langchain.vectorstores import Milvus
  milvus_connection_uri: str = ""
  # milvus_vs: VectorStore = Milvus.from_documents(
  #     texts,
  #     embeddings,
  #     collection_name="langchain_example",
  #     connection_args={"uri": milvus_connection_uri},
  # )
  return None


class VectorStoreFactory:

  def __init__(self, config: ModelConfig) -> None:
    self.config: Dict[str, Any] = config.get("vector_stores")

  def create(self, name: str) -> VectorStore:
    config: Dict[str, Any]  = self.config.get(name)
    vector_store_type: str = config.get("type")
    options: Dict[str, Any] = config.get("options")

    vector_store: VectorStore = None
    match vector_store_type:
      case "databricks":
        vector_store = databricks_vector_search(**options)
      case "chroma":
        vector_store = chroma_vector_search()
      case "milvus":
        vector_store = milvus_vector_search()
      case _:
        raise ValueError(f"Unknown vector store type: {vector_store_type}")

    return vector_store
  



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Instantiate LLM Client
# MAGIC
# MAGIC Return an OpenAI hosted gpt-4 if OPENAI_API_KEY is found in the environment. Otherwise return LLama on Databricks

# COMMAND ----------

import os

from langchain_openai import ChatOpenAI
from  langchain_core.language_models.chat_models import BaseChatModel


def databricks_llm() -> BaseChatModel:
  llm: BaseChatModel = ChatOpenAI(
    model=chat_model,
    base_url=base_url,
    api_key=api_key
  )  
  return llm


def open_ai_llm() -> BaseChatModel:
  llm: BaseChatModel = ChatOpenAI(model=chat_model)
  return llm


def get_llm() -> BaseChatModel:
    llm: BaseChatModel = None
    match "OPENAI_API_KEY" in os.environ:
        case True:
          llm = open_ai_llm()
        case _:
          llm = databricks_llm()
    return llm


llm: BaseChatModel = get_llm()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Instantiate Embeddings Client
# MAGIC
# MAGIC Return an OpenAI hosted default if OPENAI_API_KEY is found in the environment. Otherwise return GTE on Databricks

# COMMAND ----------

from langchain_core.embeddings.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from databricks_langchain import DatabricksEmbeddings

def get_embeddings() -> Embeddings:
    embeddings: Embeddings = None
    match "OPENAI_API_KEY" in os.environ:
        case True:
            embeddings = OpenAIEmbeddings()
        case _:
            embeddings: Embeddings = DatabricksEmbeddings(
                endpoint=embedding_model,
            )
    return embeddings

embeddings: Embeddings = get_embeddings()


# COMMAND ----------

import pandas as pd


generated_df: pd.DataFrame = spark.table(generated_evaluation_table_name).toPandas()
generated_df.head(3)

# COMMAND ----------

# Prepare dataframe `data` with the required format
# 
eval_df: pd.DataFrame = pd.DataFrame({})
eval_df["chunk_id"] = generated_df["chunk_id"].copy(deep=True)
eval_df["question"] = generated_df["question"].copy(deep=True)
eval_df["source"] = generated_df["source"].apply(lambda x: [x])
eval_df.head(3)

# COMMAND ----------

from langchain_core.vectorstores.base import VectorStore

chroma_vs: VectorStore = VectorStoreFactory(model_config).create(name="chroma")
print(type(chroma_vs))

# COMMAND ----------

from langchain_core.vectorstores.base import VectorStore

databricks_vs: VectorStore = VectorStoreFactory(model_config).create(name="databricks")
print(type(databricks_vs))

# COMMAND ----------

from langchain_core.vectorstores.base import VectorStore


milvus_vs: VectorStore = VectorStoreFactory(model_config).create(name="milvus")
print(type(milvus_vs))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Expected Data Format
# MAGIC
# MAGIC For each row, the retrieved relevant document IDs and the ground-truth relevant document IDs should be provided as a tuple of document ID strings.
# MAGIC
# MAGIC The column name of the retrieved relevant document IDs should be specified by the predictions parameter, and the column name of the ground-truth relevant document IDs should be specified by the targets parameter.
# MAGIC
# MAGIC ### Example Only
# MAGIC
# MAGIC This evaluation dataset can be derived from the agent evaluation synthetic data generated dataset

# COMMAND ----------

import pandas as pd


# eval_df: pd.DataFrame = pd.DataFrame(
#   {
#        "questions": [
#             "What is MLflow?",
#             "What is Databricks?",
#             "How to serve a model on Databricks?",
#             "How to enable MLflow Autologging for my workspace by default?",
#         ],
#         "retrieved_context": [
#             [
#                 "mlflow/index.html",
#                 "mlflow/quick-start.html",
#             ],
#             [
#                 "introduction/index.html",
#                 "getting-started/overview.html",
#             ],
#             [
#                 "machine-learning/model-serving/index.html",
#                 "machine-learning/model-serving/model-serving-intro.html",
#             ],
#             [],
#         ],
#         "ground_truth_context": [
#             ["mlflow/index.html"],
#             ["introduction/index.html"],
#             [
#                 "machine-learning/model-serving/index.html",
#                 "machine-learning/model-serving/llm-optimized-model-serving.html",
#             ],
#             ["mlflow/databricks-autologging.html"],
#         ],
#     } 
# )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Convert Vector Stores into chains

# COMMAND ----------

from typing import Union

from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain_core.vectorstores.base import VectorStoreRetriever


def create_retriever(retriever: Union[VectorStore, VectorStoreRetriever]) -> Chain:
    if isinstance(retriever, VectorStore):
        retriever = retriever.as_retriever(search_kwargs=search_kwargs)

    return retriever

# COMMAND ----------

from typing import List


def retrieve_doc_ids(
    question: str, 
    retriever: VectorStore, 
    doc_id_col: str = "source"
) -> List[str]:
    docs: List[Document] = retriever.invoke(question)
    return [doc.metadata[doc_id_col] for doc in docs]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC If you load the evaluation dataframe from a file you will need to using ast.literal_eval to deserialize the string representation of complex types to their appropriate types

# COMMAND ----------

import ast

# eval_df["source"] = eval_df["source"].apply(ast.literal_eval)
# eval_df["retrieved_doc_ids"] = eval_df["retrieved_doc_ids"].apply(ast.literal_eval)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Define Model Function
# MAGIC
# MAGIC This function invokes vector retrieval for each question

# COMMAND ----------

from typing import Callable, List

from functools import partial
import pandas as pd


def run_model_with(retriever: Chain) -> Callable[[pd.DataFrame], pd.DataFrame]:

  def _(input_df: pd.DataFrame):
    return input_df["question"].apply(partial(retrieve_doc_ids, retriever=retriever))

  return _


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Example of configuring Databricks as a Judge

# COMMAND ----------

from mlflow.metrics.genai import EvaluationExample, relevance

from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

llm_judge_model_name: str = "endpoints:/databricks-meta-llama-3-3-70b-instruct"
if "OPENAI_API_KEY" in os.environ:
  llm_judge_model_name = "openai:/gpt-4"

# relevance_metric = relevance(model="openai:/gpt-4")
relevance_metric = relevance(model=llm_judge_model_name)
print(relevance_metric)

# COMMAND ----------

from typing import Any, Dict, List
import mlflow

from mlflow.models.evaluation.base import (
    EvaluationMetric, 
    EvaluationResult
)

def evaluate_retriever(run_name: str, vector_store: VectorStore) -> pd.DataFrame:

  with mlflow.start_run(run_name=run_name):
    
    retriever: Chain = create_retriever(vector_store)
    
    # Consider persisting each of these datasets per retriever to avoid processing
    evaluation_with_doc_ids_pdf: pd.DataFrame = eval_df.copy()
    evaluation_with_doc_ids_pdf["retrieved_doc_ids"] = (
        evaluation_with_doc_ids_pdf["question"].apply(
            partial(retrieve_doc_ids, retriever=retriever)
        )
    )
    
    # Add additional metrics to be included in evaluation
    extra_metrics: List[EvaluationMetric] = [
        #relevance_metric,                       # Is this metric relevant?
        mlflow.metrics.latency(),
        mlflow.metrics.precision_at_k(4),
        mlflow.metrics.recall_at_k(4),
        mlflow.metrics.precision_at_k(5),
        mlflow.metrics.recall_at_k(5),
    ]

    # Column Mappings are required for the relevance_metric
    evaluator_config: Dict[str, Any] = {    
        "col_mapping": {
            "inputs": "question",
            "context": "retrieved_context",    
        }
    }

    results: EvaluationResult = mlflow.evaluate(
        run_model_with(retriever=retriever),
        eval_df,
        model_type="retriever",
        targets="source",
        predictions="retrieved_doc_ids",
        evaluators="default",
        extra_metrics=extra_metrics, # Add extra metrics here
        evaluator_config=evaluator_config,
    )
    print(results.metrics)

    return results

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Bake Off

# COMMAND ----------

from datetime import datetime
import pandas as pd

from mlflow.models.evaluation.base import EvaluationResult


time_now: datetime = datetime.now()
run_name_suffix: str = time_now.strftime("%Y%m%d%H%M%S")

bake_off: Dict[str, Any] = model_config.get("bakeoff")
competitors: List[str] = bake_off.get("competitors")

for competitor in competitors:
  competitor: str

  vector_store: VectorStore = VectorStoreFactory(model_config).create(name=competitor)
  run_name: str = f"{competitor}-{run_name_suffix}"
  evaluation_result: EvaluationResult = evaluate_retriever(run_name=run_name, vector_store=vector_store)
  eval_results_table_pdf: pd.DataFrame = evaluation_result.tables["eval_results_table"]
  display(eval_results_table_pdf)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Evaluate Chroma

# COMMAND ----------

import pandas as pd
from mlflow.models.evaluation.base import EvaluationResult

evaluation_result: EvaluationResult = evaluate_retriever(name="chroma", vector_store=chroma_vs)

eval_results_table_pdf: pd.DataFrame = evaluation_result.tables["eval_results_table"]
display(eval_results_table_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Evaluate Databricks Vector Search

# COMMAND ----------

import pandas as pd
from mlflow.models.evaluation.base import EvaluationResult

evaluation_result: EvaluationResult = evaluate_retriever(name="databricks", vector_store=databricks_vs)

eval_results_table_pdf: pd.DataFrame = evaluation_result.tables["eval_results_table"]
display(eval_results_table_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Evaluate Milvus

# COMMAND ----------

import pandas as pd
from mlflow.models.evaluation.base import EvaluationResult

evaluation_result: EvaluationResult = evaluate_retriever(name="milvus", vector_store=milvus_vs)

eval_results_table_pdf: pd.DataFrame = evaluation_result.tables["eval_results_table"]
display(eval_results_table_pdf)
