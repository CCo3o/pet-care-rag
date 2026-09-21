"""切分实验脚本（步骤 2）：对比不同 chunk_size 对检索效果的影响

运行方式（项目根目录下）：
    python chunk_experiment.py

实验设计（控制变量法）：
    - 只改 chunk_size（256 / 512 / 1024），chunk_overlap 固定 50
    - 每种切法重建一个「内存版」向量库，不碰 data/vector_store 正式索引
    - 用 14 个探测问题实测：正确文档有没有进入 top-5（命中率）+ 排名质量（MRR）
"""
import time
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma

from app.config import DOCS_DIR, EMBEDDING_MODEL
from app.ingestion import load_documents, split_documents
from app.vector_store import get_embeddings

# 探测问题集：问题 → 应该命中的文档（覆盖价目/政策/护理/应急四类）
PROBE_QUERIES = [
    ("寄养一只中型犬一天多少钱", "价目表.md"),
    ("洗澡怎么收费", "价目表.md"),
    ("临时有事去不了，取消预订会扣钱吗", "退改规则.md"),
    ("狗狗需要打什么疫苗才能寄养", "入住要求与疫苗规定.md"),
    ("兔子12小时不吃不拉怎么办", "突发疾病与受伤应急流程.md"),
    ("猫咪到新环境应激了怎么缓解", "猫咪应激管理.md"),
    ("猫可以吃巧克力吗", "猫可以吃巧克力吗.md"),
    ("几个月的小猫可以送来寄养", "幼年宠物寄养注意事项.md"),
    ("11岁的老狗寄养有什么要求", "老年宠物寄养注意事项.md"),
    ("仓鼠可以寄养吗", "异宠寄养指南.md"),
    ("春节期间怎么预订，要定金吗", "节假日预约政策.md"),
    ("寄养期间能看到我家宠物吗", "接送与探视服务.md"),
    ("自家粮食可以带来吗", "喂养与自带粮食政策.md"),
    ("每天遛几次弯", "遛弯与运动安排.md"),
]

TOP_K = 5


def evaluate(chunks, embeddings):
    """对一种切法建内存索引，跑全部探测问题，返回 (命中数, MRR, 明细)"""
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings)
    hits, mrr_total, details = 0, 0.0, []
    for query, expected in PROBE_QUERIES:
        retrieved = vs.similarity_search(query, k=TOP_K)
        rank = None
        for i, doc in enumerate(retrieved, start=1):
            if Path(doc.metadata["source"]).name == expected:
                rank = i
                break
        if rank:
            hits += 1
            mrr_total += 1 / rank
        details.append((query, expected, rank))
    return hits, mrr_total / len(PROBE_QUERIES), details


def main():
    print("📂 读取文档...")
    docs = load_documents(DOCS_DIR)
    print(f"   共 {len(docs)} 篇文档\n")

    embeddings = get_embeddings(EMBEDDING_MODEL)
    all_results = []

    for size in [256, 512, 1024]:
        chunks = split_documents(docs, chunk_size=size, chunk_overlap=50)
        print(f"🔬 chunk_size={size}：{len(chunks)} 个切片，建索引并评测中...")
        t0 = time.time()
        hits, mrr, details = evaluate(chunks, embeddings)
        cost = time.time() - t0
        all_results.append((size, len(chunks), hits, mrr, cost, details))
        print(f"   命中 {hits}/{len(PROBE_QUERIES)}，MRR={mrr:.3f}，耗时 {cost:.0f}s\n")

    # ---- 打印汇总表 ----
    print("=" * 62)
    print(f"{'chunk_size':>10} | {'切片数':>6} | {'命中(top-5)':>10} | {'MRR':>7} | {'耗时':>8}")
    print("-" * 62)
    for size, n, hits, mrr, cost, _ in all_results:
        print(f"{size:>10} | {n:>6} | {hits:>10}/{len(PROBE_QUERIES)} | {mrr:>7.3f} | {cost:>6.0f}s")
    print("=" * 62)

    # ---- 落盘实验报告 ----
    report = [f"# 切分实验结果（{datetime.now():%Y-%m-%d %H:%M}）\n",
              f"文档 {len(docs)} 篇，探测问题 {len(PROBE_QUERIES)} 条，overlap 固定 50，top_k={TOP_K}\n",
              "| chunk_size | 切片数 | 命中(top-5) | MRR | 耗时 |", "|---|---|---|---|---|"]
    for size, n, hits, mrr, cost, _ in all_results:
        report.append(f"| {size} | {n} | {hits}/{len(PROBE_QUERIES)} | {mrr:.3f} | {cost:.0f}s |")

    report.append("\n## 各问题命中明细（✓=命中名次，✗=未进 top-5）\n")
    report.append("| 问题 | 期望文档 | " + " | ".join(f"{s}" for s, *_ in [(r[0],) for r in all_results]) + " |")
    report.append("|---|---|" + "---|" * len(all_results))
    for qi, (query, expected) in enumerate(PROBE_QUERIES):
        cells = []
        for (_, _, _, _, _, details) in all_results:
            rank = details[qi][2]
            cells.append(f"✓{rank}" if rank else "✗")
        report.append(f"| {query} | {expected} | " + " | ".join(cells) + " |")

    out = Path("docs") / "切分实验结果.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\n📄 报告已保存：{out}")


if __name__ == "__main__":
    main()
