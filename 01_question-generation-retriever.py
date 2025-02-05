# Databricks notebook source
# MAGIC %pip install --quiet --upgrade beautifulsoup4 langchain openai pandas seaborn scikit-learn python-dotenv
# MAGIC %restart_python

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

import json
import os

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
from bs4 import BeautifulSoup
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Cache:
    def __init__(self, persist_path, cache_loading_fn):
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


def cached_openai_ChatCompletion_create(**kwargs):
    cache = kwargs.pop("cache")
    return cache.get_from_cache_or_load_cache(**kwargs)


def embeddings_embed_documents_fn(**kwargs):
    chunk = kwargs.get("chunk")
    return embeddings.embed_documents([chunk])


def cached_langchain_openai_embeddings(**kwargs):
    cache = kwargs.pop("cache")
    return cache.get_from_cache_or_load_cache(**kwargs)


# COMMAND ----------

# Other configurations

# Choose a seed for reproducible results
SEED = 2023

# For cost-saving purposes, choose a path to persist the responses for LLM calls
CACHE_PATH = "_cache.json"
EMBEDDINGS_CACHE_PATH = "_embeddings_cache.json"

# To avoid re-running the scraping process, choose a path to save the scrapped docs
SCRAPPED_DATA_PATH = "mlflow_docs_scraped.csv"

# Choose a path to save the generated dataset
OUTPUT_DF_PATH = "question_answer_source.csv"


# COMMAND ----------

cache = Cache(CACHE_PATH, chat_completion_create_fn)
embeddings_cache = Cache(EMBEDDINGS_CACHE_PATH, embeddings_embed_documents_fn)


# COMMAND ----------

CHUNK_SIZE = 1500


# COMMAND ----------

page = requests.get("https://mlflow.org/docs/latest/index.html")
soup = BeautifulSoup(page.content, "html.parser")

mainLocation = "https://mlflow.org/docs/latest/"
header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}

data = []
for a_link in soup.find_all("a"):
    document_url = mainLocation + a_link["href"]
    page = requests.get(document_url, headers=header)
    soup = BeautifulSoup(page.content, "html.parser")
    file_to_store = a_link.get("href")
    if soup.find("div", {"class": "rst-content"}):
        data.append(
            [
                file_to_store,
                soup.find("div", {"class": "rst-content"}).text.replace("\n", " "),
            ]
        )

df = pd.DataFrame(data, columns=["source", "text"])

display(df)

# COMMAND ----------

df.to_csv(SCRAPPED_DATA_PATH, index=False)
df = pd.read_csv(SCRAPPED_DATA_PATH)


# COMMAND ----------

# For demonstration purposes, let's pick 5 popular MLflow documantation pages from the dataset
mask = df["source"].isin(
    {
        "tracking.html",
        "models.html",
        "model-registry.html",
        "search-runs.html",
        "projects.html",
    }
)
sub_df = df[mask]

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, separator=" ")


def get_chunks(input_row):
    new_rows = []
    chunks = text_splitter.split_text(input_row["text"])
    for i, chunk in enumerate(chunks):
        new_rows.append({"chunk": chunk, "source": input_row["source"], "chunk_index": i})
    return new_rows


expanded_df = pd.DataFrame(columns=["chunk", "source", "chunk_index"])

for index, row in sub_df.iterrows():
    new_rows = get_chunks(row)
    expanded_df = pd.concat([expanded_df, pd.DataFrame(new_rows)], ignore_index=True)

expanded_df.head(3)


# COMMAND ----------

# For cost-saving purposes, let's pick the first 3 chunks from each doc
# To generate questions with more chunks, change the start index and end index in iloc[]
start, end = 0, 3
filtered_df = (
    expanded_df.groupby("source").apply(lambda x: x.iloc[start:end]).reset_index(drop=True)
)
filtered_df.head(3)


# COMMAND ----------

filtered_df["chunk"][0]


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

    return cached_openai_ChatCompletion_create(
        messages=messages,
        model="gpt-4o-mini",
        functions=[submit_function],
        function_call="auto",
        temperature=0.0,
        seed=SEED,
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

queries = []


# COMMAND ----------

get_raw_response(filtered_df["chunk"][0])



# COMMAND ----------

# The requests sometimes get ratelimited, you can re-execute this cell without losing the existing results.
n = len(filtered_df)
for i, row in filtered_df.iterrows():
    chunk = row["chunk"]
    question, answer = generate_question_answer(chunk)
    print(f"{i+1}/{n}: {question}")
    queries.append(
        {
            "question": question,
            "answer": answer,
            "chunk": chunk,
            "chunk_id": row["chunk_index"],
            "source": row["source"],
        }
    )


# COMMAND ----------

result_df = pd.DataFrame(queries)
result_df = result_df[result_df["answer"] != "N/A"]


# COMMAND ----------

def add_to_output_df(result_df=pd.DataFrame({})):
    """
    This function adds the records in result_df to the existing records saved at OUTPUT_DF_PATH,
    remove the duplicate rows and save the new collection of records back to OUTPUT_DF_PATH.
    """
    if os.path.exists(OUTPUT_DF_PATH):
        all_result_df = pd.read_csv(OUTPUT_DF_PATH)
    else:
        all_result_df = pd.DataFrame({})
    all_result_df = (
        pd.concat([all_result_df, result_df], ignore_index=True)
        .drop_duplicates()
        .sort_values(by=["source", "chunk_id"])
        .reset_index(drop=True)
    )
    all_result_df.to_csv(OUTPUT_DF_PATH, index=False)
    return all_result_df


# COMMAND ----------

all_result_df = add_to_output_df(result_df)


# COMMAND ----------

all_result_df.head(3)


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
questions = all_result_df["question"].to_list()
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
tsne = TSNE(n_components=2, random_state=SEED)
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

all_result_df.sample(3)


# COMMAND ----------

embedded_queries = all_result_df.copy()
embedded_queries["chunk_emb"] = all_result_df["chunk"].apply(
    lambda x: np.squeeze(cached_langchain_openai_embeddings(chunk=x, cache=embeddings_cache))
)
embedded_queries["question_emb"] = all_result_df["question"].apply(
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
