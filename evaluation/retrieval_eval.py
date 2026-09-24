"""检索质量评测：只验证“是否找对资料”，不调用 DeepSeek API。

运行：
    python evaluation/retrieval_eval.py

输出：
    docs/检索评测报告.md
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# 允许从项目根目录直接运行：python evaluation/retrieval_eval.py。
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.config import CHROMA_DIR, DOCS_DIR, EMBEDDING_MODEL, TOP_K
from app.ingestion import load_documents
from app.vector_store import get_embeddings, get_vector_store


DEFAULT_CASES_PATH = Path(__file__).with_name("retrieval_cases.json")
DEFAULT_REPORT_PATH = PROJECT_DIR / "docs" / "检索评测报告.md"


def load_cases(path: Path) -> list[dict]:
    """读取并校验评测题，尽早发现错别字或不存在的来源文件。"""
    cases = json.loads(path.read_text(encoding="utf-8"))
    required_fields = {"id", "category", "question", "expected_sources"}

    available_sources = {
        Path(doc.metadata["source"]).name for doc in load_documents(DOCS_DIR)
    }
    for case in cases:
        missing_fields = required_fields - case.keys()
        if missing_fields:
            raise ValueError(f"评测题缺少字段 {missing_fields}: {case}")
        if not case["expected_sources"]:
            raise ValueError(f"评测题没有标注期望来源: {case['id']}")
        unknown_sources = set(case["expected_sources"]) - available_sources
        if unknown_sources:
            raise ValueError(
                f"评测题 {case['id']} 引用了不存在或已归档的来源: {unknown_sources}"
            )
    return cases


def evaluate(vector_store, cases: list[dict], k: int) -> list[dict]:
    """计算每题是否命中、首个命中名次和实际检索到的文件名。"""
    results = []
    for case in cases:
        retrieved_docs = vector_store.similarity_search(case["question"], k=k)
        retrieved_sources = [Path(doc.metadata["source"]).name for doc in retrieved_docs]
        rank = next(
            (
                position
                for position, source in enumerate(retrieved_sources, start=1)
                if source in case["expected_sources"]
            ),
            None,
        )
        results.append({**case, "rank": rank, "retrieved_sources": retrieved_sources})
    return results


def calculate_metrics(results: list[dict]) -> tuple[float, float]:
    """返回 Recall@k 与 MRR@k。未命中的题目对 MRR 贡献为 0。"""
    hit_count = sum(result["rank"] is not None for result in results)
    mrr = sum(1 / result["rank"] for result in results if result["rank"]) / len(results)
    return hit_count / len(results), mrr


def render_report(results: list[dict], k: int, chunk_count: int) -> str:
    """将结果渲染成适合提交到作品集的 Markdown 报告。"""
    recall, mrr = calculate_metrics(results)
    grouped_results = defaultdict(list)
    for result in results:
        grouped_results[result["category"]].append(result)

    lines = [
        f"# 检索评测报告（{datetime.now():%Y-%m-%d %H:%M}）",
        "",
        "> 本报告只测检索是否召回正确资料，不调用 DeepSeek，也不评价最终回答措辞。",
        "",
        f"- 评测题数：{len(results)}",
        f"- 检索参数：top-k = {k}",
        f"- 当前索引切片数：{chunk_count}",
        f"- Recall@{k}：{recall:.1%}（{sum(result['rank'] is not None for result in results)}/{len(results)}）",
        f"- MRR@{k}：{mrr:.3f}",
        "",
        "## 按类别统计",
        "",
        "| 类别 | 题数 | Recall | MRR |",
        "|---|---:|---:|---:|",
    ]
    for category, category_results in grouped_results.items():
        category_recall, category_mrr = calculate_metrics(category_results)
        lines.append(
            f"| {category} | {len(category_results)} | {category_recall:.1%} | {category_mrr:.3f} |"
        )

    missed_results = [result for result in results if result["rank"] is None]
    lines.extend(["", "## 未命中问题", ""])
    if not missed_results:
        lines.append(f"全部问题均命中 top-{k}。")
    else:
        lines.extend([
            "| ID | 问题 | 期望来源 | 实际 top-k 来源 |",
            "|---|---|---|---|",
        ])
        for result in missed_results:
            expected = "、".join(result["expected_sources"])
            actual = "、".join(result["retrieved_sources"])
            lines.append(f"| {result['id']} | {result['question']} | {expected} | {actual} |")

    lines.extend([
        "",
        "## 使用说明",
        "",
        "- 先修改资料或检索代码，再执行本脚本；不要只比较不同运行之间的单条问答感受。",
        "- 对未命中题，先确认资料是否写清楚、期望来源是否标对，再考虑切分、混合检索或重排序。",
        "- 该评测集是第一版基线；后续应继续加入同义改写、多轮省略问句和拒答案例。",
    ])
    return "\n".join(lines) + "\n"


def main():
    # Windows PowerShell may use GBK by default; keep report output printable.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="宠物寄养 RAG 检索评测（不调用 LLM）")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="评测题 JSON 文件")
    parser.add_argument("--k", type=int, default=TOP_K, help="检索 top-k")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="输出 Markdown 报告路径")
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k 必须大于 0")

    cases = load_cases(args.cases)
    embeddings = get_embeddings(EMBEDDING_MODEL)
    vector_store = get_vector_store(embeddings, CHROMA_DIR)
    chunk_count = vector_store._collection.count()
    if chunk_count == 0:
        raise RuntimeError("索引为空，请先执行 python -m app.main --rebuild")

    results = evaluate(vector_store, cases, args.k)
    report = render_report(results, args.k, chunk_count)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    recall, mrr = calculate_metrics(results)
    hit_count = sum(result["rank"] is not None for result in results)
    print(f"✅ 评测完成：{hit_count}/{len(results)} 命中 top-{args.k}")
    print(f"   Recall@{args.k}={recall:.1%}，MRR@{args.k}={mrr:.3f}")
    print(f"📄 报告已保存：{args.report}")


if __name__ == "__main__":
    main()
