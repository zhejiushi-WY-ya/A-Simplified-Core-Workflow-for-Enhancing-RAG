import json, re


def safe_json(text: str):
    if not text:
        return {}

    text = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(text)
    except:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass

    return {}
