from openai import OpenAI

client = OpenAI(
    api_key="sk-eGYT382xngt2u4kGGnxInmjYvqloG8ltr07UbSKvo7w2uBI7",
    base_url="https://rtekkxiz.bja.sealos.run/v1",
)

try:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # 不行就换成你服务端实际支持的模型
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
    )
    print("调用成功，账号大概率有余额")
    print(resp.choices[0].message.content)
except Exception as e:
    print("调用失败：")
    print(e)
    
