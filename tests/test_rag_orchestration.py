"""Tests for RAG orchestration layer."""

from src.rag.orchestrator import KnowledgeOrchestrator, RetrievalResult


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, results: list[RetrievalResult]):
        self.results = results
        self.query_count = 0

    def query(self, text: str, game_id: str) -> list[RetrievalResult]:
        """Return mock results."""
        self.query_count += 1
        return self.results


class FailingRetriever:
    """Retriever that always raises an exception."""

    def query(self, text: str, game_id: str) -> list[RetrievalResult]:
        """Always raises an exception."""
        raise RuntimeError("Retriever failure")


def test_orchestrator_initialization():
    """Test orchestrator can be initialized."""
    orchestrator = KnowledgeOrchestrator()
    assert orchestrator is not None


def test_retriever_registration():
    """Test retrievers can be registered."""
    orchestrator = KnowledgeOrchestrator()
    retriever = MockRetriever([])
    
    orchestrator.register_retriever(retriever)
    assert len(orchestrator._retrievers) == 1


def test_orchestrator_merges_results():
    """Test orchestrator merges results from multiple retrievers."""
    orchestrator = KnowledgeOrchestrator()
    
    retriever1 = MockRetriever([
        RetrievalResult(content="Result 1", source="source1", confidence=0.9)
    ])
    retriever2 = MockRetriever([
        RetrievalResult(content="Result 2", source="source2", confidence=0.8)
    ])
    
    orchestrator.register_retriever(retriever1)
    orchestrator.register_retriever(retriever2)
    
    results = orchestrator.resolve("test query", "test-game")
    
    assert len(results) == 2
    assert retriever1.query_count == 1
    assert retriever2.query_count == 1


def test_orchestrator_sorts_by_confidence():
    """Test orchestrator sorts results by confidence descending."""
    orchestrator = KnowledgeOrchestrator()
    
    retriever = MockRetriever([
        RetrievalResult(content="Low", source="s1", confidence=0.5),
        RetrievalResult(content="High", source="s2", confidence=0.9),
        RetrievalResult(content="Medium", source="s3", confidence=0.7),
    ])
    
    orchestrator.register_retriever(retriever)
    results = orchestrator.resolve("test query", "test-game")
    
    assert len(results) == 3
    assert results[0].confidence == 0.9
    assert results[1].confidence == 0.7
    assert results[2].confidence == 0.5


def test_orchestrator_isolates_failures():
    """Test orchestrator isolates failures from individual retrievers."""
    orchestrator = KnowledgeOrchestrator()
    
    # Register a failing retriever and a working retriever
    orchestrator.register_retriever(FailingRetriever())
    working_retriever = MockRetriever([
        RetrievalResult(content="Working", source="source", confidence=0.8)
    ])
    orchestrator.register_retriever(working_retriever)
    
    results = orchestrator.resolve("test query", "test-game")
    
    # Should get results from working retriever despite failing retriever
    assert len(results) == 1
    assert results[0].content == "Working"


def test_orchestrator_handles_empty_results():
    """Test orchestrator handles retrievers that return no results."""
    orchestrator = KnowledgeOrchestrator()
    
    retriever = MockRetriever([])
    orchestrator.register_retriever(retriever)
    
    results = orchestrator.resolve("test query", "test-game")
    
    assert len(results) == 0
    assert retriever.query_count == 1
