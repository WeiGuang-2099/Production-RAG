"""Prompts for the agent's control nodes (router, grader, rewriter, etc.)."""

ROUTER_PROMPT = """Classify how to handle the user's question. Reply with exactly one word:
- retrieve: needs information looked up in the document corpus
- answer: a greeting or general question that needs no document lookup
- clarify: too vague or ambiguous to act on

Question: {question}

One word:"""

GRADER_PROMPT = """Decide whether the retrieved context is sufficient to answer the question.
Reply with exactly one word: yes or no.

Question: {question}

Context:
{context}

Sufficient (yes/no):"""

REWRITE_PROMPT = """The previous search did not retrieve enough relevant context. Rewrite the
question into a better search query using keywords and key entities. Return only the query.

Question: {question}

Rewritten search query:"""

ANSWER_DIRECTLY_PROMPT = """Answer the user's question directly and concisely. This is a general
question not grounded in the document corpus, so do not invent citations.

Question: {question}

Answer:"""

CLARIFY_PROMPT = """The user's question is too vague to answer well. Ask one concise clarifying
question to narrow it down.

Question: {question}

Clarifying question:"""
