from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    if not documents:
        return []

    # CHUNK_SIZE / CHUNK_OVERLAP are token counts (as documented), so measure
    # length with the tiktoken encoder rather than raw characters. This keeps
    # chunks within the embedding model's token budget instead of being ~4x
    # smaller than intended. disallowed_special=() avoids errors if a document
    # happens to contain text like "<|endoftext|>".
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        disallowed_special=(),
    )

    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks
