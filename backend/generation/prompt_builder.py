from backend.chat.schemas import SourceDoc

SYSTEM_PROMPT = (
     """You are a precise question-answering assistant.

STRICT RULES:
1. Answer ONLY the exact question asked. Nothing more.
2. Use ONLY information from the current context chunk provided.
3. Do NOT mention any other policies or topics not asked about.
4. Keep your answer to 1-2 sentences maximum.
5. End with: Source: [filename], page [number].
6. If the context does not contain the answer, say "I don't know."

EXAMPLE:
Question: What is the sick leave policy?
Answer: Employees may take up to 10 days of paid sick leave per year, with a medical certificate required for absences exceeding 3 consecutive days. Source: sample.txt, page 0.

DO NOT add any extra information beyond what was asked."""
)


def build_prompt(
    query: str,
    context_docs: list[SourceDoc],
    history: list[dict],
) -> list[dict]:
    relevant_docs = [d for d in context_docs if d.score > 0.3]
    if not relevant_docs:
        relevant_docs = sorted(context_docs, key=lambda d: d.score, reverse=True)[:1]

    context_blocks = []
    for i, doc in enumerate(relevant_docs, 1):
        page_info = f", page {doc.page}" if doc.page is not None else ""
        context_blocks.append(
            f"--- Chunk {i} (Source: {doc.source}{page_info}) ---\n{doc.content}"
        )

    context_text = "\n\n".join(context_blocks) if context_blocks else "No context available."

    system_content = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}"

    messages: list[dict] = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": (
        f"Question: {query}\n"
        "Answer ONLY this specific question using ONLY the context provided above.\n"
        "Answer:"
    )}
]

    return messages