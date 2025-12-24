import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dotenv import load_dotenv
from jinja2 import Template
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm

from mem0 import Memory
from src.memzero.add import get_mem0_config
from src.utils import get_llm_client

load_dotenv()


class MemorySearch:
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False):
        # 使用本地 mem0，配置 DeepSeek 和 Qwen
        config = get_mem0_config(is_graph)
        self.mem0_client = Memory.from_config(config)

        self.top_k = top_k
        self.llm_client = get_llm_client()
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph

        # 性能统计
        self.timing_stats = defaultdict(list)

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        timing_breakdown = {}
        start_time = time.time()

        retries = 0
        while retries < max_retries:
            try:
                # 本地 mem0 Memory 类的 search 方法
                # 内部包含: query embedding + vector search
                t1 = time.time()
                memories = self.mem0_client.search(
                    query, user_id=user_id, limit=self.top_k
                )
                t2 = time.time()
                timing_breakdown["mem0_search_api_time"] = t2 - t1
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        # 结果后处理计时
        t1 = time.time()
        semantic_memories = []
        for memory in memories.get("results", memories) if isinstance(memories, dict) else memories:
            mem_data = {
                "memory": memory.get("memory", ""),
                "timestamp": memory.get("metadata", {}).get("timestamp", ""),
                "score": round(memory.get("score", 0), 2),
            }
            semantic_memories.append(mem_data)

        graph_memories = None
        if self.is_graph and isinstance(memories, dict) and "relations" in memories:
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories["relations"]
            ]
        t2 = time.time()
        timing_breakdown["result_processing_time"] = t2 - t1

        end_time = time.time()
        timing_breakdown["total_search_time"] = end_time - start_time
        timing_breakdown["num_memories_returned"] = len(semantic_memories)

        return semantic_memories, graph_memories, end_time - start_time, timing_breakdown

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        timing_breakdown = {}

        # Search speaker 1 memories
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time, speaker_1_timing = self.search_memory(
            speaker_1_user_id, question
        )
        timing_breakdown["speaker_1_search"] = speaker_1_timing

        # Search speaker 2 memories
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time, speaker_2_timing = self.search_memory(
            speaker_2_user_id, question
        )
        timing_breakdown["speaker_2_search"] = speaker_2_timing

        # Prompt construction
        t1 = time.time()
        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )
        t2 = time.time()
        timing_breakdown["prompt_construction_time"] = t2 - t1
        timing_breakdown["prompt_length_chars"] = len(answer_prompt)

        # LLM API call
        t1 = time.time()
        response = self.llm_client.chat.completions.create(
            model=os.getenv("MODEL"), messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
        )
        t2 = time.time()
        timing_breakdown["llm_api_time"] = t2 - t1

        # Token usage (if available)
        if hasattr(response, "usage") and response.usage:
            timing_breakdown["input_tokens"] = response.usage.prompt_tokens
            timing_breakdown["output_tokens"] = response.usage.completion_tokens
            timing_breakdown["total_tokens"] = response.usage.total_tokens

        response_time = timing_breakdown["llm_api_time"]
        timing_breakdown["total_response_time"] = response_time

        return (
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            timing_breakdown,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            timing_breakdown,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        # 记录统计信息
        self.timing_stats["speaker_1_memory_time"].append(speaker_1_memory_time)
        self.timing_stats["speaker_2_memory_time"].append(speaker_2_memory_time)
        self.timing_stats["response_time"].append(response_time)
        self.timing_stats["total_search_time"].append(speaker_1_memory_time + speaker_2_memory_time)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
            # 详细耗时分解
            "timing_breakdown": timing_breakdown,
        }

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
            ):
                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                self.results[idx].append(result)

                # Save results after each question is processed
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        # 打印性能统计摘要
        self._print_timing_summary()

    def _print_timing_summary(self):
        """打印并保存性能统计摘要"""
        summary = {
            "technique": "Mem0",
            "config": {
                "top_k": self.top_k,
                "filter_memories": self.filter_memories,
                "is_graph": self.is_graph,
                "model": os.getenv("MODEL"),
                "embedding_model": os.getenv("EMBEDDING_MODEL"),
            },
            "metrics": {},
            "time_distribution": {},
        }

        print("\n" + "=" * 60)
        print("Mem0 Performance Summary")
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
        if self.timing_stats["total_search_time"] and self.timing_stats["response_time"]:
            total_search = sum(self.timing_stats["total_search_time"])
            total_response = sum(self.timing_stats["response_time"])
            total = total_search + total_response

            speaker_1_total = sum(self.timing_stats["speaker_1_memory_time"])
            speaker_2_total = sum(self.timing_stats["speaker_2_memory_time"])

            summary["time_distribution"] = {
                "speaker_1_search_total_seconds": round(speaker_1_total, 2),
                "speaker_1_search_percentage": round(100 * speaker_1_total / total, 1),
                "speaker_2_search_total_seconds": round(speaker_2_total, 2),
                "speaker_2_search_percentage": round(100 * speaker_2_total / total, 1),
                "total_search_seconds": round(total_search, 2),
                "total_search_percentage": round(100 * total_search / total, 1),
                "response_total_seconds": round(total_response, 2),
                "response_percentage": round(100 * total_response / total, 1),
                "total_seconds": round(total, 2),
            }

            print(f"\nTime Distribution:")
            print(f"  Speaker 1 Search: {speaker_1_total:.2f}s ({100*speaker_1_total/total:.1f}%)")
            print(f"  Speaker 2 Search: {speaker_2_total:.2f}s ({100*speaker_2_total/total:.1f}%)")
            print(f"  Total Search:     {total_search:.2f}s ({100*total_search/total:.1f}%)")
            print(f"  Response:         {total_response:.2f}s ({100*total_response/total:.1f}%)")
            print(f"  Total:            {total:.2f}s")

        print("=" * 60 + "\n")

        # 保存到文件
        stats_file_path = self.output_path.replace(".json", "_performance_stats.json")
        with open(stats_file_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Performance stats saved to: {stats_file_path}")

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results
