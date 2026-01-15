
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

  # .master("local[*]") \
spark = SparkSession.builder.appName("da231o_assignment_1") \
  .master("spark://localhost:7077") \
  .config("spark.executor.memory", "1g") \
  .config("spark.driver.memory", "1g") \
  .config("spark.memory.offHeap.enabled", "true") \
  .config("spark.memory.offHeap.size", "1g") \
  .getOrCreate()
  
# NOTE: This code snippet will be replaced during evaluation with the relevant sample input files.
INPUT_DF_SIZE = "tiny"
# INPUT_DF_SIZE = "short"
# INPUT_DF_SIZE = "medium"
# INPUT_DF_SIZE = "large"

keyword_df = spark.read.csv("/data/keywords.csv", header=True)
keyword_df.show()

translate_df = spark.read.csv("/data/translate.csv", header=True)
translate_df.show()

stopword_df = spark.read.csv("/data/stopwords.csv", header=True)
stopword_df.show()

input_df = spark.read.parquet("/data/cc_tiny.parquet")
input_df.show()

print(input_df.count())
print(keyword_df.count())
print(translate_df.count())
print(stopword_df.count())

print(input_df.printSchema())
print(keyword_df.printSchema())
print(translate_df.printSchema())
print(stopword_df.printSchema())

#######################################
###!@0.4 START COMMON USER IMPORTS
#######################################
## Specify valid imports, if any, for ALL your answers  ==========
## start your edits here =================
import math
import time
from sentence_transformers import SentenceTransformer

import re
from pyspark.sql.types import StringType, LongType, DoubleType, ArrayType, MapType
from pyspark.sql.functions import udf
from pyspark.sql.window import Window
from pyspark import SparkContext

## end your edits here =================
###!@0.4 END COMMON USER IMPORTS

#######################################
###!@0.5 START COMMON USER FUNCTIONS
#######################################
## Specify user defined functions, if any, used by multiple answers   =====

def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    global model
    if "model" not in globals():
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(chunks).tolist()

def cosine_similarity(u, v):
    dot = sum(ui * vi for ui, vi in zip(u, v))
    norm_u = math.sqrt(sum(ui * ui for ui in u))
    norm_v = math.sqrt(sum(vi * vi for vi in v))
    denom = norm_u * norm_v
    return dot / denom if denom != 0 else 0.0

## start your edits here =================

@udf(ArrayType(DoubleType()))
def generate_embeddings_single(chunk: str) -> list[float]:
    global model
    if "model" not in globals():
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode([chunk]).tolist()[0]

@udf(ArrayType(ArrayType(DoubleType())))
def generate_embeddings_list(chunks: list[str]) -> list[list[float]]:
    global model
    if "model" not in globals():
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(chunks).tolist()

@udf(DoubleType())
def similarity_score_topic(topic_emb: list[float], chunks_emb: list[list[float]]):
    score_sum = 0
    for chunk_emb in chunks_emb:
        dot = sum(ui * vi for ui, vi in zip(topic_emb, chunk_emb))
        norm_u = math.sqrt(sum(ui * ui for ui in topic_emb))
        norm_v = math.sqrt(sum(vi * vi for vi in chunk_emb))
        denom = norm_u * norm_v
        score_sum += dot / denom if denom != 0 else 0.0
    return score_sum

def create_regex_from_list(list_str):
  escaped_strings = [re.escape(s) for s in list_str]
  return '|'.join(escaped_strings)

@udf(StringType())
def remove_stopwords(content: str, stopwords_list: list[str], punctuation_pattern: str):
    return "".join(["" if word in stopwords_list else word for word in re.split(punctuation_pattern, content)])

@udf(MapType(StringType(), LongType()))
def count_keywords_month(keywords: list[str], contents: list[str]):
  keyword_counts = {keyword: 0 for keyword in keywords}

  for content in contents:
    for keyword in keywords:
      if keyword in content:
        keyword_counts[keyword] += 1

  return keyword_counts

@udf(MapType(StringType(), LongType()))
def count_keywords_month_lang(keywords: list[str], contents: list[str]):
  keyword_counts = {keyword: 0 for keyword in keywords}

  for keyword in keywords:
    pattern = rf"\b{re.escape(keyword)}\b"
    for content in contents:
        if re.search(pattern, content):
            keyword_counts[keyword] += 1

  return keyword_counts

## end your edits here =================

def question_1_1(input_df):

  ## start your edits here =================

  output_df = input_df.withColumn("crawl_month", F.substring(F.col("date"), 0, 7))

  ## end your edits here =================

  return output_df

def question_1_2(input_df, stopword_df):

  ## start your edits here =================

  escape_seq = ['\\n', '\\t', '\\r', '\\b', '\\f', '\\a', '\\\\', '\\"', "\\'"]
  escape_seq_reg_expr = create_regex_from_list(escape_seq)
  stopwords_list = stopword_df.select("stopwords").rdd.flatMap(lambda x: x).collect()
  punctuations = [".", ",", "!", "?", ";", ":", "(", ")", "{", "}", "[", "]", "'", '"', "<", ">", "/", " "]
  punctuation_pattern = "([" + "".join(re.escape(c) for c in punctuations) + "])"

  sc = SparkContext.getOrCreate()
  punctuation_pattern = sc.broadcast(punctuation_pattern)
  stopwords_list = sc.broadcast(stopwords_list)

  input_df = input_df.drop("date").withColumn("content", F.lower(F.col("content"))) #lowercase
  input_df = input_df.withColumn("content", F.regexp_replace(F.col("content"), r'<.*?>', " ")) #HTML Tags
  input_df = input_df.withColumn("content", F.regexp_replace(F.col("content"), escape_seq_reg_expr, " ")) # Escape Seqs
  input_df = input_df.withColumn("content", F.trim(F.regexp_replace(F.col("content"), r'\s+', ' '))) # Whitespaces
  en_input_df = input_df.where(F.col("lang") == "en")
  en_input_df = en_input_df.withColumn("content", remove_stopwords(F.col("content"), F.lit(stopwords_list.value), F.lit(punctuation_pattern.value))) # Stopwords
  input_df = input_df.where(F.col("lang") != "en").union(en_input_df)

  output_df = input_df.select("uri", "lang", "crawl_month", "content")

  ## end your edits here =================

  return output_df

def question_2_1(input_df):

  ## start your edits here =================

  output_df = input_df.where(F.col("lang") == "en")

  ## end your edits here =================

  return output_df

def question_2_2(input_df, keyword_df):

  ## start your edits here =================

  keyword_list = keyword_df.withColumn("keyword", F.lower("keyword")).select("keyword").distinct().rdd.flatMap(lambda x: x).collect()

  input_df = input_df.groupBy("crawl_month").agg(F.count("*").alias("total_docs"),F.collect_list("content").alias("contents")).withColumnRenamed("crawl_month", "month")
  input_df = input_df.withColumn("keywords", F.lit(keyword_list))
  input_df = input_df.withColumn("doc_count_map", count_keywords_month(F.col("keywords"), F.col("contents"))).drop("contents")
  input_df = input_df.select("month", "total_docs", F.explode("doc_count_map").alias("keyword", "doc_count"))
  input_df = input_df.withColumn("normalized_doc_count", F.try_divide("doc_count", "total_docs")).drop("total_docs")

  output_df = input_df.select("keyword", "month", "doc_count", "normalized_doc_count")

  ## end your edits here =================

  return output_df

def question_3_1(input_df):

  ## start your edits here =================

  months_of_interest = ["2025-04", "2025-05", "2025-06", "2025-07"]

  output_df = input_df.filter(F.col("month").isin(months_of_interest)).groupBy("keyword").agg(F.avg(F.col("normalized_doc_count")).alias("avg_normalized_doc_count"))
  quantiles = output_df.approxQuantile("avg_normalized_doc_count", [0.25, 0.50, 0.75, 1.0], 0)

  output_df = output_df.withColumn("25_percentile", F.col("avg_normalized_doc_count") == 0)
  output_df = output_df.withColumn("50_percentile", (F.col("avg_normalized_doc_count") > 0) & (F.col("avg_normalized_doc_count") <= quantiles[1]))
  output_df = output_df.withColumn("75_percentile", (F.col("avg_normalized_doc_count") > quantiles[1]) & (F.col("avg_normalized_doc_count") <= quantiles[2]))
  output_df = output_df.withColumn("100_percentile", (F.col("avg_normalized_doc_count") > quantiles[2]) & (F.col("avg_normalized_doc_count") <= quantiles[3]))

  ## end your edits here =================

  return output_df

def question_3_2(input_df):

  ## start your edits here =================

  months_order = input_df.select("month").distinct().sort("month").rdd.flatMap(lambda x: x).collect()
  min_month = months_order[0]

  normalized_prev = input_df.where(F.col("month") == min_month).drop("doc_count", "month").withColumnRenamed("normalized_doc_count", "prev_normal")
  output_df = normalized_prev.drop("prev_normal").withColumn("month", F.lit(min_month)).withColumn("ROC", F.lit(0))

  for month in months_order[1:]:
    normalized_curr = input_df.where(F.col("month") == month).drop("doc_count", "month").withColumnRenamed("normalized_doc_count", "curr_normal")
    joined_df = normalized_prev.join(normalized_curr, "keyword", "inner").withColumn("month", F.lit(month))
    joined_df = joined_df.withColumn("ROC", F.try_divide(F.try_subtract("curr_normal", "prev_normal"), "prev_normal"))
    output_df = output_df.union(joined_df.drop("prev_normal", "curr_normal"))
    normalized_prev = normalized_curr.withColumnRenamed("curr_normal", "prev_normal")

  output_df = output_df.fillna({"ROC": 0}).withColumn("trending_flag", F.col("ROC") >= 1)

  ## end your edits here =================

  return output_df

def question_3_3(input_df):

  ## start your edits here =================

    windowSpec = Window.partitionBy("keyword").orderBy("month")
    input_df = input_df.withColumn("prevROC", F.lag(F.col("ROC"), 1, default=0).over(windowSpec))
    input_df = input_df.withColumn("roc_diff_enough", F.when(F.col("ROC") - F.col("prevROC") < 0.5, False).otherwise(True))
    input_df = input_df.groupBy("keyword").agg(F.min("roc_diff_enough").alias("min"), F.max("roc_diff_enough").alias("max"))
    output_df = input_df.withColumn("emerging_flag", F.when((F.col("min") == True) & (F.col("max") == True), True).otherwise(False)).drop("min", "max")

  ## end your edits here =================

    return output_df

def question_4_1(input_df, translate_df):

  langs = ["ja", "es", "ru"]

  keyword_langs_list = {
      "ja": translate_df.select(F.lower("Japanese")).distinct().rdd.flatMap(lambda x: x).collect(),
      "es": translate_df.select(F.lower("Spanish")).distinct().rdd.flatMap(lambda x: x).collect(),
      "ru": translate_df.select(F.lower("Russian")).distinct().rdd.flatMap(lambda x: x).collect()
  }

  output_df = None
  for lang in langs:
      input_df_lang = input_df.filter(F.col("lang") == lang)

      docs_per_month = input_df_lang.groupBy("crawl_month", "lang").agg(F.count("*").alias("total_docs"), F.collect_list("content").alias("contents")).withColumnRenamed("crawl_month", "month")
      input_df_lang = docs_per_month.withColumn("keywords", F.lit(keyword_langs_list[lang]))
      input_df_lang = input_df_lang.withColumn("doc_count_map", count_keywords_month_lang(F.col("keywords"), F.col("contents"))).drop("contents")
      input_df_lang = input_df_lang.select("month", "total_docs", "lang", F.explode("doc_count_map").alias("keyword", "doc_count"))
      input_df_lang = input_df_lang.withColumn("normalized_doc_count", F.try_divide("doc_count", "total_docs")).drop("total_docs", "doc_count")

      if output_df is None:
          output_df = input_df_lang
      else:
          output_df = input_df_lang.union(output_df)

  output_df = output_df.select("lang", "keyword", "month", "normalized_doc_count")

  ## end your edits here =================

  return output_df

def question_4_2(input_df):

  ## start your edits here =================

  months_of_interest = ["2025-04", "2025-05", "2025-06", "2025-07"]

  roc_df = input_df.where(F.col("month") == months_of_interest[0]).drop("normalized_doc_count").withColumn("ROC", F.lit(0))
  normalized_prev = input_df.where(F.col("month") == months_of_interest[0]).drop("month").withColumnRenamed("normalized_doc_count", "prev_normal")

  for month in months_of_interest[1:]:
    normalized_curr = input_df.where(F.col("month") == month).drop("month").withColumnRenamed("normalized_doc_count", "curr_normal")
    joined_df = normalized_prev.join(normalized_curr, ["lang", "keyword"], "inner").withColumn("month", F.lit(month))
    joined_df = joined_df.withColumn("ROC", F.try_divide(F.try_subtract("curr_normal", "prev_normal"), "prev_normal"))
    roc_df = roc_df.union(joined_df.drop("prev_normal", "curr_normal"))
    normalized_prev = normalized_curr.withColumnRenamed("curr_normal", "prev_normal")

  roc_df = roc_df.fillna({"ROC": 0})

  windowSpec = Window.partitionBy("lang", "keyword").orderBy("month")
  roc_df = roc_df.withColumn("prevROC", F.lag(F.col("ROC"), 1, default=0).over(windowSpec))
  roc_df = roc_df.withColumn("roc_diff_enough", F.when(F.col("ROC") - F.col("prevROC") < 0.5, False).otherwise(True))
  roc_df = roc_df.groupBy("lang", "keyword").agg(F.min("roc_diff_enough").alias("min"), F.max("roc_diff_enough").alias("max"))
  output_df = roc_df.withColumn("emerging_flag", F.when((F.col("min") == True) & (F.col("max") == True), True).otherwise(False)).drop("min", "max")

  ## end your edits here =================

  return output_df

def question_4_3(input_df_eng_lang, input_df_3_lang, translate_df):

  ## start your edits here =================

  emerging_en_df = input_df_eng_lang.withColumn("lang", F.lit("en")).select("lang", "keyword", "emerging_flag")
  emerging_full_df = emerging_en_df.union(input_df_3_lang)
  emerging_true_full = emerging_full_df.filter(F.col("emerging_flag") == True).drop("emerging_flag")

  lang_map = {
      "es": "Spanish",
      "ru": "Russian",
      "ja": "Japanese"
  }

  rows = emerging_true_full.filter(F.col("lang") == "en").select("keyword").rdd.flatMap(lambda x: x).collect()
  for lang in ["es", "ja", "ru"]:
      rows.extend(emerging_true_full.filter(F.col("lang") == lang).select("keyword").withColumnRenamed("keyword", lang_map[lang]).join(translate_df.select(lang_map[lang], "English"), on=lang_map[lang]).select("English").rdd.flatMap(lambda x: x).collect())

  output_df = spark.createDataFrame([(row,) for row in rows], "globally_emerging_topics: string").distinct()

  ## end your edits here =================

  return output_df

def question_5_1(input_df):

  ## start your edits here =================

  chunk_length = 256
  max_chunks = 5
  output_df = input_df.withColumn("total_chunks", F.ceil(F.try_divide(F.length(F.col("content")), F.lit(chunk_length))))

  columns = []
  for i in range(max_chunks):
      columns.append(f"chunk_{i}")
      output_df = output_df.withColumn(f"chunk_{i}", F.substr(F.col("content"), F.try_add(F.try_multiply(F.floor(F.try_divide(F.try_multiply(F.lit(i), F.col("total_chunks")), F.lit(max_chunks))), F.lit(chunk_length)), F.lit(1)), F.lit(chunk_length)))

  output_df = output_df.withColumn("chunks", F.array_distinct(F.array(columns))).drop(*columns, "total_chunks", "lang", "content", "crawl_month")

  ## end your edits here =================

  return output_df


def question_5_2(input_df, globally_emerging_topics_df):

  ## start your edits here =================

  # Reduce input_df size to 10 rows
  # input_df = input_df.limit(10)

  dummy_topics = ["AI", "5G", "sustainable development"]
  if globally_emerging_topics_df.count() < 3:
      globally_emerging_topics_df = spark.createDataFrame([(topic, ) for topic in dummy_topics], schema="globally_emerging_topics: string")

  topics_emb = globally_emerging_topics_df.distinct().withColumn("topic_emb", generate_embeddings_single(F.col("globally_emerging_topics")))
  uri_chunks_emb = input_df.withColumn("chunks_emb", generate_embeddings_list(F.col("chunks"))).drop("chunks")

  topics_emb_map = {row["globally_emerging_topics"]: row["topic_emb"] for row in topics_emb.collect()}
  topics_emb_map_str_list = [f"{key}:{topics_emb_map[key]}" for key in topics_emb_map.keys()]

  sc = SparkContext.getOrCreate()
  topics_emb_map_str_list = sc.broadcast(topics_emb_map_str_list)
  uri_chunks_emb = uri_chunks_emb.withColumn("topic_emb_map_str", F.lit(topics_emb_map_str_list.value))
  # print("Input DF before splitting column: ", uri_chunks_emb.collect())
  topic_emb_split_col = F.split("topic_emb_map", ":")
  # print("Input DF before error: ", uri_chunks_emb.collect())
  # uri_chunks_emb = uri_chunks_emb.withColumn("topic_emb_map", F.explode("topic_emb_map_str"))
  # uri_chunks_emb = uri_chunks_emb.withColumn("topic", F.split("topic_emb_map", ":").getItem(0)).withColumn("topic_emb", F.split("topic_emb_map", ":").getItem(1))
  uri_chunks_emb = uri_chunks_emb.select("*", F.explode("topic_emb_map_str").alias("topic_emb_map"), topic_emb_split_col.getItem(0).alias("topic"), topic_emb_split_col.getItem(1).alias("topic_emb"))
  uri_chunks_emb = uri_chunks_emb.withColumn("topic_emb", F.from_json(F.col("topic_emb"), ArrayType(DoubleType()))).drop("topic_emb_map", "topic_emb_map_str")
  uri_topic_score = uri_chunks_emb.withColumn("score", similarity_score_topic(F.col("topic_emb"), F.col("chunks_emb"))).drop("topic_emb", "chunks_emb")

  windowSpec = Window.partitionBy("topic").orderBy(F.col("score").desc())
  ranked_df = uri_topic_score.withColumn("rank", F.rank().over(windowSpec))
  output_df = ranked_df.filter(F.col("rank") <= 5).sort(F.col("rank")).drop("score")
  output_df = output_df.select("topic", "rank", "uri")

  ## end your edits here =================

  return output_df


###!@6.1 START SANITY CHECK OF DATA TYPES
# ========== *** DO NOT MODIFY *** ========== #

from pyspark.sql import DataFrame

def get_reference_df(question: str, input_data_size: str) -> DataFrame:
    # This functions returns the correct dataframe for a given question and dataset size
    mapping = {
        ("ans1_1", "tiny"):  "ans1_1_REF_tiny_output_df.parquet",
        ("ans1_1", "small"): "ans1_1_REF_small_output_df.parquet",
        ("ans1_2", "tiny"):  "ans1_2_REF_tiny_output_df.parquet",
        ("ans1_2", "small"): "ans1_2_REF_small_output_df.parquet",
        ("ans2_1", "tiny"):  "ans2_1_REF_tiny_output_df.parquet",
        ("ans2_1", "small"): "ans2_1_REF_small_output_df.parquet",
        ("ans2_2", "tiny"):  "ans2_2_REF_tiny_output_df.parquet",
        ("ans2_2", "small"): "ans2_2_REF_small_output_df.parquet",
        ("ans3_1", "tiny"):  "ans3_1_REF_tiny_output_df.parquet",
        ("ans3_1", "small"): "ans3_1_REF_small_output_df.parquet",
        ("ans3_2", "tiny"):  "ans3_2_REF_tiny_output_df.parquet",
        ("ans3_2", "small"): "ans3_2_REF_small_output_df.parquet",
        ("ans3_3", "tiny"):  "ans3_3_REF_tiny_output_df.parquet",
        ("ans3_3", "small"): "ans3_3_REF_small_output_df.parquet",
        ("ans4_1", "tiny"):  "ans4_1_REF_tiny_output_df.parquet",
        ("ans4_1", "small"): "ans4_1_REF_small_output_df.parquet",
        ("ans4_2", "tiny"):  "ans4_2_REF_tiny_output_df.parquet",
        ("ans4_2", "small"): "ans4_2_REF_small_output_df.parquet",
        ("ans4_3", "tiny"):  "ans4_3_REF_tiny_output_df.parquet",
        ("ans4_3", "small"): "ans4_3_REF_small_output_df.parquet",
        ("ans5_1", "tiny"):  "ans5_1_REF_tiny_output_df.parquet",
        ("ans5_1", "small"): "ans5_1_REF_small_output_df.parquet",
        ("ans5_2", "tiny"):  "ans5_2_REF_tiny_output_df.parquet",
        ("ans5_2", "small"): "ans5_2_REF_small_output_df.parquet",
    }

    try:
        path = mapping[(question, input_data_size)]
    except KeyError:
        raise Exception(f"Invalid question={question} and dataset size={input_data_size}")

    base_path = "/data/reference_outputs/"

    return spark.read.parquet(base_path+path)


def checkDF(question: str, input_data_size: str, user_output: DataFrame) -> bool:
    ref_df = get_reference_df(question, input_data_size)

    ref_schema = [(f.name, f.dataType) for f in ref_df.schema.fields]
    user_schema = [(f.name, f.dataType) for f in user_output.schema.fields]

    # Schema check
    if ref_schema != user_schema:
        print("Schema mismatch")
        print("Expected schema: ")
        ref_df.printSchema()
        print("Your schema:")
        user_output.printSchema()
        return False

    # Number of Records check
    if ref_df.count() != user_output.count():
        print("Number of Records mismatch")
        return False

    # Difference in row content check (ignoring the order of records)
    diff1 = ref_df.subtract(user_output).count()
    diff2 = user_output.subtract(ref_df).count()

    if diff1 == 0 and diff2 == 0:
        return True
    else:
        print("Row content mismatch")
        return False

###!@6.1 END SANITY CHECK OF DATA TYPES

ref_output_5_1 = spark.read.parquet("/data/reference_outputs/ans5_1_REF_tiny_output_df.parquet")
ref_output_4_3 = spark.read.parquet("/data/reference_outputs/ans4_3_REF_tiny_output_df.parquet")

ans5_2_output_df = question_5_2(ref_output_5_1, ref_output_4_3)
# Print physical plan
ans5_2_output_df.explain()
print('answer 5_2: ', checkDF("ans5_2", INPUT_DF_SIZE, ans5_2_output_df))

# ref_output_5_2 = spark.read.parquet("reference_outputs/ans5_2_REF_tiny_output_df.parquet")
# diff_in_my_df = ans5_2_output_df.exceptAll(ref_output_5_2)
# print("Rows present in your ans5_2_output_df but not in ref_output_5_2:")
# diff_in_my_df.show(truncate=False)

# diff_in_ref_df = ref_output_5_2.exceptAll(ans5_2_output_df)
# print("Rows present in ref_output_5_2 but not in your ans5_2_output_df:")
# diff_in_ref_df.show(truncate=False)