# from openai import OpenAI
#
# client = OpenAI(
#     base_url="http://localhost:18889/v1",
#     api_key=''
# )
#
# resp = client.chat.completions.create(
#     model="Qwen3-30B-A3B-Instruct-2507",
#     messages=[
#         {"role": "user", "content": "你好，请只回复：测试成功"}
#     ],
# )
#
# print(resp.choices[0].message.content)


from openai import OpenAI


BASE_URL = "http://127.0.0.1:18889/v1"   # 改成你的远程 embedding 地址
API_KEY = ""                   # 改成你的 key
MODEL = "Qwen3-Embedding-0.6B"           # 改成你的模型名


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    texts = [
        "苹果是一种水果",
        "香蕉也是一种水果",
        "今天天气很好，适合出去散步",
    ]

    res = client.embeddings.create(
        model=MODEL,
        input=texts,
    )

    vectors = [x.embedding for x in res.data]

    print("请求成功")
    print("返回条数:", len(vectors))
    print("向量维度:", len(vectors[0]))

    sim_01 = cosine_similarity(vectors[0], vectors[1])
    sim_02 = cosine_similarity(vectors[0], vectors[2])

    print(f"“苹果是一种水果” vs “香蕉也是一种水果” 相似度: {sim_01:.4f}")
    print(f"“苹果是一种水果” vs “今天天气很好，适合出去散步” 相似度: {sim_02:.4f}")


if __name__ == "__main__":
    main()
