import json, re
from json_repair import repair_json


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
            try:
                return json.loads(repair_json(m.group(), skip_json_loads=True))
            except:
                pass

    try:
        return json.loads(repair_json(text, skip_json_loads=True))
    except:
        pass

    return {}
