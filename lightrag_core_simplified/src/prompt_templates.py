from lightrag.prompt import PROMPTS


DEFAULT_TUPLE_DELIMITER = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
DEFAULT_COMPLETION_DELIMITER = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
KEYWORDS_EXTRACTION_EXAMPLES = "\n".join(PROMPTS["keywords_extraction_examples"])


GRAPH_ENTITY_TYPES = [
    "Person",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Content",
    "Artifact",
    "NaturalObject",
    "Other",
]


INDEX_EXTRACTION_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Goal---
Extract grounded entities and meaningful relationships from the input text.

---Instructions---
1. Identify clearly defined and meaningful entities in the text.
2. For each entity, return:
   - `name`: consistent entity name
   - `type`: one of {entity_types}
   - `description`: concise but comprehensive description grounded only in the text
3. Identify direct, clearly stated, meaningful relationships between extracted entities.
4. If a statement implies an n-ary relation, decompose it into reasonable binary relations.
5. Treat relationships as undirected unless direction is explicit. Do not output duplicates.
6. For each relation, return:
   - `source`
   - `target`
   - `keywords`: short high-level keywords
   - `description`: concise grounded explanation
7. Use consistent naming across entities and relations.
8. Avoid pronouns in entity and relation descriptions.
9. Output only valid JSON in the exact schema below.

---Output JSON Schema---
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "Entity Type",
      "description": "Grounded description"
    }}
  ],
  "relations": [
    {{
      "source": "Source Entity",
      "target": "Target Entity",
      "keywords": ["keyword1", "keyword2"],
      "description": "Grounded relationship description"
    }}
  ]
}}

---Input Text---
{input_text}
"""


KEYWORDS_EXTRACTION_PROMPT = PROMPTS["keywords_extraction"]


def build_keywords_extraction_prompt(query, language):
    base_prompt = KEYWORDS_EXTRACTION_PROMPT.format(
        query=query,
        language=language,
        examples=KEYWORDS_EXTRACTION_EXAMPLES,
    )
    return """---Additional Retrieval Guidance---
Extract keywords that maximize recall for retrieval.

1. Include the main topic, target entities, events, methods, locations, time ranges, constraints, and domain terms from the query.
2. Include plausible alternate phrasings, aliases, abbreviations, and near-synonyms when they are likely to appear in source text.
3. Put broad intents and themes into `high_level_keywords`.
4. Put concrete names, terms, and highly specific constraints into `low_level_keywords`.
5. Prefer retrieval-friendly noun phrases over full sentences.
6. Output only the required JSON schema.

""" + base_prompt


CONTEXT_COMPRESSION_PROMPT = """---Role---
You are a Knowledge Graph Specialist, proficient in evidence selection and answer planning.

---Task---
Compress the retrieval context into a question-focused evidence brief for downstream question answering.

---Question---
{query}

---Instructions---
1. Focus only on evidence that helps answer the question directly and completely.
2. Integrate relevant evidence from entities, relations, supporting facts, and document chunks.
3. Preserve important names, numbers, dates, causes, conditions, comparisons, and explicit relationships.
4. Merge overlapping evidence, but do not lose distinct aspects of the answer.
5. If the question has multiple parts, create evidence for each part.
6. If the context contains limitations, uncertainty, or missing evidence, state that clearly.
7. Keep every statement grounded in the provided context. Do not add outside knowledge.
8. Output plain text only in the following sections:
   Answer Focus:
   Key Evidence:
   Missing or Uncertain Points:
   Reference Hints:

---Context---
{context}
"""


DESCRIPTION_SUMMARY_PROMPT = PROMPTS["summarize_entity_descriptions"]


def build_description_summary_prompt(description_type, description_name, description_list):
    return DESCRIPTION_SUMMARY_PROMPT.format(
        description_type=description_type,
        description_name=description_name,
        description_list=description_list,
        summary_length=256,
        language="the same language as the source descriptions",
    )


ANSWER_PROMPT = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided Context.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph Data, Supporting Facts, and Document Chunks found in the Context.

---Instructions---

1. Carefully determine the user's exact information need and the likely sub-questions behind it.
2. Scrutinize the Context, especially the Knowledge Graph Data, Supporting Facts, Document Chunks, and Compressed Reasoning Notes.
3. Start with a direct answer in the first paragraph.
4. Then cover all major relevant aspects needed for a complete answer, not just the single most obvious point.
5. Prefer concrete facts, named entities, explicit relationships, and document-grounded details.
6. When the question is asking for explanation, comparison, cause, process, impact, or recommendation, structure the answer to cover each needed dimension explicitly.
7. If the evidence supports multiple complementary points, include them together rather than collapsing to one.
8. If the Context is incomplete or ambiguous, state what is supported, what is uncertain, and where the gap is.
9. Do not invent, assume, or infer information not explicitly supported by the Context.
10. Use Markdown for readability.
11. Be helpful to the reader: explain why the facts matter or how they answer the question, but stay grounded in the Context.
12. If useful, use a short bullet list after the opening paragraph to improve clarity and coverage.
13. End with a `### References` section that cites the most relevant chunk references using the provided reference ids.
14. Do not output anything after the references section.

---Context---
{context_data}

---Question---
{query}
"""
