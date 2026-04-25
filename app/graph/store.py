class GraphStore:
    """Graph store for managing entity relationships."""

    def get_neighbors(self, entity: str, depth: int = 1) -> list[tuple[str, str]]:
        raise NotImplementedError
