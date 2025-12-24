import json
import os
import time
from collections import defaultdict

import numpy as np
import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from tqdm import tqdm

from src.utils import get_embedding_client, get_llm_client

load_dotenv()

PROMPT = """
# Question: 
{{QUESTION}}

# Context: 
{{CONTEXT}}

# Short answer:
"""


class RAGManager:
    def __init__(self, data_path="dataset/locomo10_rag.json", chunk_size=500, k=1):
        self.model = os.getenv("MODEL")
        self.llm_client = get_llm_client()
        self.embedding_client = get_embedding_client()
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.k = k
        # 性能统计
        self.timing_stats = defaultdict(list)

    def generate_response(self, question, context):
        timing_breakdown = {}

        # Step 1: Prompt construction
        t1 = time.time()
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)
        t2 = time.time()
        timing_breakdown["prompt_construction_time"] = t2 - t1

        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                # Step 2: LLM API call
                t1 = time.time()
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can answer "
                            "questions based on the provided context."
                            "If the question involves timing, use the conversation date for reference."
                            "Provide the shortest possible answer."
                            "Use words directly from the conversation when possible."
                            "Avoid using subjects in your answer.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                t2 = time.time()
                timing_breakdown["llm_api_time"] = t2 - t1

                # 提取 token 使用信息（如果可用）
                if hasattr(response, "usage") and response.usage:
                    timing_breakdown["input_tokens"] = response.usage.prompt_tokens
                    timing_breakdown["output_tokens"] = response.usage.completion_tokens
                    timing_breakdown["total_tokens"] = response.usage.total_tokens

                timing_breakdown["total_response_time"] = (
                    timing_breakdown["prompt_construction_time"] + timing_breakdown["llm_api_time"]
                )

                return response.choices[0].message.content.strip(), timing_breakdown["total_response_time"], timing_breakdown
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                time.sleep(1)  # Wait before retrying

    def clean_chat_history(self, chat_history):
        cleaned_chat_history = ""
        for c in chat_history:
            cleaned_chat_history += f"{c['timestamp']} | {c['speaker']}: {c['text']}\n"

        return cleaned_chat_history

    def calculate_embedding(self, document):
        response = self.embedding_client.embeddings.create(model=os.getenv("EMBEDDING_MODEL"), input=document)
        return response.data[0].embedding

    def calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def search(self, query, chunks, embeddings, k=1):
        """
        Search for the top-k most similar chunks to the query.

        Args:
            query: The query string
            chunks: List of text chunks
            embeddings: List of embeddings for each chunk
            k: Number of top chunks to return (default: 1)

        Returns:
            combined_chunks: The combined text of the top-k chunks
            search_time: Time taken for the search
            timing_breakdown: Dict with detailed timing for each step
        """
        timing_breakdown = {}
        t_start = time.time()

        # Step 1: Query Embedding
        t1 = time.time()
        query_embedding = self.calculate_embedding(query)
        t2 = time.time()
        timing_breakdown["query_embedding_time"] = t2 - t1

        # Step 2: Calculate similarities
        t1 = time.time()
        similarities = [self.calculate_similarity(query_embedding, embedding) for embedding in embeddings]
        t2 = time.time()
        timing_breakdown["similarity_calculation_time"] = t2 - t1

        # Step 3: Top-K selection
        t1 = time.time()
        if k == 1:
            top_indices = [np.argmax(similarities)]
        else:
            top_indices = np.argsort(similarities)[-k:][::-1]
        t2 = time.time()
        timing_breakdown["topk_selection_time"] = t2 - t1

        # Step 4: Combine chunks
        t1 = time.time()
        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])
        t2 = time.time()
        timing_breakdown["chunk_combination_time"] = t2 - t1

        t_end = time.time()
        timing_breakdown["total_search_time"] = t_end - t_start

        return combined_chunks, t_end - t_start, timing_breakdown

    def create_chunks(self, chat_history, chunk_size=500):
        """
        Create chunks using tiktoken for more accurate token counting
        """
        # 使用通用的 cl100k_base encoding，兼容非 OpenAI 模型
        try:
            encoding = tiktoken.encoding_for_model(os.getenv("EMBEDDING_MODEL"))
        except KeyError:
            # 如果模型不被 tiktoken 识别（如 Qwen），使用通用 encoding
            encoding = tiktoken.get_encoding("cl100k_base")

        documents = self.clean_chat_history(chat_history)

        if chunk_size == -1:
            return [documents], []

        chunks = []

        # Encode the document
        tokens = encoding.encode(documents)

        # Split into chunks based on token count
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk = encoding.decode(chunk_tokens)
            chunks.append(chunk)

        embeddings = []
        for chunk in chunks:
            embedding = self.calculate_embedding(chunk)
            embeddings.append(embedding)

        return chunks, embeddings

    def process_all_conversations(self, output_file_path):
        with open(self.data_path, "r") as f:
            data = json.load(f)

        FINAL_RESULTS = defaultdict(list)
        for key, value in tqdm(data.items(), desc="Processing conversations"):
            chat_history = value["conversation"]
            questions = value["question"]

            chunks, embeddings = self.create_chunks(chat_history, self.chunk_size)

            for item in tqdm(questions, desc="Answering questions", leave=False):
                question = item["question"]
                answer = item.get("answer", "")
                category = item["category"]

                if self.chunk_size == -1:
                    context = chunks[0]
                    search_time = 0
                    search_timing = {"total_search_time": 0, "note": "full_context_mode"}
                else:
                    context, search_time, search_timing = self.search(question, chunks, embeddings, k=self.k)

                response, response_time, response_timing = self.generate_response(question, context)

                # 记录统计信息
                self.timing_stats["search_time"].append(search_time)
                self.timing_stats["response_time"].append(response_time)
                if "query_embedding_time" in search_timing:
                    self.timing_stats["query_embedding_time"].append(search_timing["query_embedding_time"])

                FINAL_RESULTS[key].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "context": context,
                        "response": response,
                        "search_time": search_time,
                        "response_time": response_time,
                        # 详细耗时分解
                        "timing_breakdown": {
                            "search": search_timing,
                            "response": response_timing,
                        },
                    }
                )
                with open(output_file_path, "w+") as f:
                    json.dump(FINAL_RESULTS, f, indent=4)

        # Save results
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)

        # 打印并保存性能统计摘要
        self._print_timing_summary(output_file_path)

    def _print_timing_summary(self, output_file_path):
        """打印并保存性能统计摘要"""
        summary = {
            "technique": "RAG",
            "config": {
                "chunk_size": self.chunk_size,
                "k": self.k,
                "model": self.model,
                "embedding_model": os.getenv("EMBEDDING_MODEL"),
            },
            "metrics": {},
            "time_distribution": {},
        }

        print("\n" + "=" * 60)
        print("RAG Performance Summary")
        print("=" * 60)

        for key, values in self.timing_stats.items():
            if values:
                avg = np.mean(values)
                std = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                p50 = np.percentile(values, 50)
                p95 = np.percentile(values, 95)
                p99 = np.percentile(values, 99)

                summary["metrics"][key] = {
                    "average": round(avg, 4),
                    "std_dev": round(std, 4),
                    "min": round(float(min_val), 4),
                    "max": round(float(max_val), 4),
                    "p50": round(p50, 4),
                    "p95": round(p95, 4),
                    "p99": round(p99, 4),
                    "count": len(values),
                    "total": round(sum(values), 4),
                }

                print(f"\n{key}:")
                print(f"  Average: {avg:.4f}s")
                print(f"  Std Dev: {std:.4f}s")
                print(f"  Min:     {min_val:.4f}s")
                print(f"  Max:     {max_val:.4f}s")
                print(f"  P50:     {p50:.4f}s")
                print(f"  P95:     {p95:.4f}s")
                print(f"  P99:     {p99:.4f}s")
                print(f"  Count:   {len(values)}")

        # 计算总耗时占比
        if self.timing_stats["search_time"] and self.timing_stats["response_time"]:
            total_search = sum(self.timing_stats["search_time"])
            total_response = sum(self.timing_stats["response_time"])
            total = total_search + total_response

            summary["time_distribution"] = {
                "search_total_seconds": round(total_search, 2),
                "search_percentage": round(100 * total_search / total, 1),
                "response_total_seconds": round(total_response, 2),
                "response_percentage": round(100 * total_response / total, 1),
                "total_seconds": round(total, 2),
            }

            print(f"\nTime Distribution:")
            print(f"  Search:   {total_search:.2f}s ({100*total_search/total:.1f}%)")
            print(f"  Response: {total_response:.2f}s ({100*total_response/total:.1f}%)")
            print(f"  Total:    {total:.2f}s")

        print("=" * 60 + "\n")

        # 保存到文件
        stats_file_path = output_file_path.replace(".json", "_performance_stats.json")
        with open(stats_file_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Performance stats saved to: {stats_file_path}")
