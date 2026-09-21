"""快速验证 DeepSeek API Key 是否有效（30 秒测试）"""
from langchain_openai import ChatOpenAI
from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def main():
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-xxxx"):
        print("❌ Key 未配置或还是占位符，请检查 .env 文件")
        return

    print(f"🔌 正在连接 DeepSeek（模型: {DEEPSEEK_MODEL}）...")
    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        temperature=0.3,
    )
    resp = llm.invoke("用一句话回答：猫能吃巧克力吗？")
    print(f"✅ 连接成功！DeepSeek 回复：{resp.content}")


if __name__ == "__main__":
    main()
