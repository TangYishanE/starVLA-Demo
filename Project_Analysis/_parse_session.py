import json, sys

path = r"C:\Users\TangYishan\AppData\Local\hermes\cache\spillover\call_00_s8tRql7Lcy6cXDtl5TKk6289.txt"
d = json.load(open(path, encoding="utf-8"))
msgs = d.get("messages", [])
print("total messages:", len(msgs))
print("=" * 80)
# print the last 14 messages: role + text (and whether assistant had tool_calls)
for m in msgs[-14:]:
    role = m.get("role")
    content = m.get("content")
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict):
                parts.append(x.get("text", ""))
            else:
                parts.append(str(x))
        content = " ".join(parts)
    content = (content or "").strip()
    tc = m.get("tool_calls")
    has_tc = bool(tc)
    print(f"\n----- [{role}] (id={m.get('id')}, tool_calls={has_tc}) -----")
    print((content or "<empty>")[:2000])
    if tc:
        for t in tc:
            fn = t.get("function", {})
            print("   >> tool:", fn.get("name"), "|", (fn.get("arguments") or "")[:300])
