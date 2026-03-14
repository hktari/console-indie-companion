"""Connection pool for QMD MCP clients to avoid repeated process spawning."""

import logging
import threading
from typing import Optional

from src.rag.qmd_client import QmdMcpStdioClient, QmdQueryResult

logger = logging.getLogger(__name__)


class QmdConnectionPool:
    """Thread-safe connection pool for QMD MCP clients.
    
    Maintains persistent connections per index to avoid the overhead
    of spawning new processes for each query.
    """

    def __init__(self):
        self._clients: dict[str, QmdMcpStdioClient] = {}
        self._lock = threading.Lock()

    def get_client(self, index_name: str) -> QmdMcpStdioClient:
        """Get or create a client for the given index.
        
        Args:
            index_name: QMD index name
            
        Returns:
            Persistent QMD client for this index
        """
        with self._lock:
            if index_name not in self._clients:
                logger.debug("Creating new QMD client for index: %s", index_name)
                self._clients[index_name] = QmdMcpStdioClient(index_name)
            return self._clients[index_name]

    def query(
        self, text: str, game_id: str, index_name: str = "game-companion", limit: int = 5
    ) -> list[QmdQueryResult]:
        """Query using a pooled client.
        
        Args:
            text: Query text
            game_id: Game/collection identifier
            index_name: QMD index name
            limit: Maximum number of results
            
        Returns:
            List of query results
        """
        client = self.get_client(index_name)
        return client.query(text, game_id, limit)

    def shutdown(self):
        """Shutdown all pooled clients."""
        with self._lock:
            for index_name, client in self._clients.items():
                logger.debug("Shutting down pooled client for index: %s", index_name)
                try:
                    client.shutdown()
                except Exception as e:
                    logger.warning(
                        "Error shutting down client for %s: %s", index_name, e
                    )
            self._clients.clear()

    def __del__(self):
        """Cleanup on garbage collection."""
        self.shutdown()


# Global connection pool instance
_global_pool: Optional[QmdConnectionPool] = None
_pool_lock = threading.Lock()


def get_qmd_pool() -> QmdConnectionPool:
    """Get the global QMD connection pool."""
    global _global_pool
    with _pool_lock:
        if _global_pool is None:
            _global_pool = QmdConnectionPool()
        return _global_pool
