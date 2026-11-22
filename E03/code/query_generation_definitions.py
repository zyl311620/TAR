import json


def messages(task: str, text: str, recipe: str):
    language = "Japanese" if task.endswith("-ja") else "English"
    relation = (
        "Given an amendment rationale, retrieve existing pre-revision provisions that may need amendment."
        if task.startswith("rat") else
        "Given the visible query article, retrieve other provisions that may need co-amendment."
    )
    instructions = {
        "q2e": "Expand this query with relevant legal terms and close paraphrases for retrieval. "
               "Be concise: at most 100 words or 300 Japanese characters. Do not repeat items.",
        "q2d": "Write a hypothetical existing legal provision useful for retrieving revision candidates. "
               "Do not invent statute numbers, citations, dates or article identifiers. "
               "Use at most 200 words or 500 Japanese characters.",
        "task_q2e": relation + " Extract search terms about the existing legal mechanism, "
                    "scope, actors, conditions and dependencies. Use at most 120 words "
                    "or 350 Japanese characters; do not repeat terms.",
        "multi5": relation + " Produce exactly five short, distinct search queries with the same "
                  "revision-retrieval intent. Each should emphasize a different visible aspect. "
                  "Do not infer an unseen before/after change. Each query must be concise: "
                  "at most 75 words or 200 Japanese characters. No repetitions within a query.",
        "long_single": relation + " Write one extended retrieval query of about 500–650 tokens, "
                       "preserving the visible facts and giving complementary search terminology.",
    }
    schema = '{"queries":["query 1","query 2","query 3","query 4","query 5"]}' if recipe == "multi5" else (
        '{"query":"one retrieval text"}')
    return [
        {"role": "system", "content":
         f"Write in {language}. Use only the supplied query as a factual source. "
         "Preserve actors, negation, conditions, exceptions and explicit citations. "
         "Do not add relevance labels, guessed target IDs, or external legal facts. "
         "Return only the specified JSON object; no reasoning or Markdown."},
        {"role": "user", "content": instructions[recipe] + "\nOutput schema: " + schema +
         "\n\nComplete source query:\n" + text},
    ]


def parse_generation(content: str, recipe: str):
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No complete JSON object")
    value = json.loads(content[start:end+1])
    if recipe == "multi5":
        queries = value.get("queries")
        if (not isinstance(queries, list) or len(queries) != 5
                or any(not isinstance(q, str) or not q.strip() for q in queries)):
            raise ValueError("Expected exactly five nonempty query strings")
        return {"queries": [q.strip() for q in queries]}
    query = value.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Expected nonempty query text")
    return {"query": query.strip()}

