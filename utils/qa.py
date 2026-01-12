import re
from utils.ollama_client import ollama_generate


def answer_question(question, vectorstore, mode, model="phi3:mini"):
    q = (question or "").strip()
    if not q:
        return ("Please enter a question.", [])

    # ✅ Normalize question (removes punctuation + collapses spaces)
    q_norm = re.sub(r"[^a-zA-Z0-9\s]", " ", q.lower())
    q_norm = re.sub(r"\s+", " ", q_norm).strip()

    # ✅ Stronger summary intent detection
    # Works for: "summarize ?", "summary", "give me a summary", "overview", etc.
    SUMMARY_KEYWORDS = [
        "summary", "summarize", "overview", "high level", "high level summary",
        "what is this document about", "what is this handbook about",
        "give me a summary", "summarize this document", "summarize the document"
    ]
    is_summary = any(k in q_norm for k in SUMMARY_KEYWORDS)

    # ✅ Extra: if the whole question is basically just "summarize"
    if q_norm in {"summarize", "summary", "overview"}:
        is_summary = True

    # Retrieve with similarity scores
    results = vectorstore.similarity_search_with_score(q, k=8)

    SIMILARITY_THRESHOLD = 0.8

    if is_summary:
        # For summary: broad coverage (do NOT threshold)
        relevant_docs = [doc for doc, _score in results][:6]
    else:
        # For Q&A: thresholded + strict
        relevant_docs = [doc for doc, score in results if score < SIMILARITY_THRESHOLD][:3]

    if not relevant_docs:
        return ("The requested information is not available in the provided documents.", [])

    # Build context (cap for speed)
    context = "\n\n---\n\n".join(d.page_content.strip() for d in relevant_docs)
    context = context[:3500] if is_summary else context[:2500]

    # Style
    if mode == "Plain":
        style = (
            "Write in simple plain language. Keep it short (6-10 bullet points max)."
        )
    else:
        style = (
            "Write a structured answer with headings and bullet points. "
            "Keep it readable for non-technical staff. Avoid long paragraphs."
        )

    # Task
    if is_summary:
        task = (
            "Create a clear summary of the document using ONLY the excerpts. "
            "Cover: purpose, key policies/processes, roles/responsibilities, and any important rules. "
            "Do not add assumptions."
        )
        not_found_rule = (
            "If the excerpts are empty or unrelated, say: "
            "\"Not found in the provided documents.\""
        )
    else:
        task = (
            "Answer the user’s question using ONLY the excerpts. "
            "Do not add assumptions."
        )
        not_found_rule = (
            "If the answer is not clearly supported by the excerpts, respond exactly with: "
            "\"Not found in the provided documents.\""
        )

    prompt = f"""
You are an internal document assistant.

{task}

{not_found_rule}

{style}

User request:
{q}

Document excerpts:
{context}
""".strip()

    # Generate (local Ollama)
    answer = ollama_generate(prompt=prompt, model=model, temperature=0.2)

    # If Ollama returns a warning message, fallback to excerpts
    if answer.startswith("⚠️"):
        fallback = answer + "\n\nMost relevant excerpts:\n\n"
        if mode == "Plain":
            fallback += "\n\n".join(d.page_content.strip()[:350] for d in relevant_docs)
        else:
            fallback += "\n\n".join(d.page_content.strip() for d in relevant_docs)

        sources = [
            f"{d.metadata.get('source')} (page/section {d.metadata.get('page', d.metadata.get('section'))})"
            for d in relevant_docs
        ]
        return fallback, sources

    # If model says not found, return consistent message
    if "not found in the provided documents" in answer.lower():
        return ("The requested information is not available in the provided documents.", [])

    # Sources (dedupe)
    sources = []
    seen = set()
    for d in relevant_docs:
        src = f"{d.metadata.get('source')} (page/section {d.metadata.get('page', d.metadata.get('section'))})"
        if src not in seen:
            sources.append(src)
            seen.add(src)

    return answer, sources
