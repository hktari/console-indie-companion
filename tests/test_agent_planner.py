"""Tests for agent planner routing logic."""

from src.agent.models import RouteType
from src.agent.planner import RequestPlanner


class TestRequestPlanner:
    """Test request planner routing decisions."""

    def test_memory_query_routing(self):
        """Test that memory-related queries are routed to memory search."""
        planner = RequestPlanner(qmd_url=None, model="gpt-4.1-mini")

        # Test various memory-related phrases
        memory_queries = [
            "what did I do earlier?",
            "where was that item I saw before?",
            "remember when I fought that boss?",
            "what happened last time?",
        ]

        for query in memory_queries:
            decision = planner.plan(query, None, "")
            assert decision.route == RouteType.MEMORY_SEARCH
            assert "memory_search" in decision.tools_to_call

    def test_web_research_routing(self):
        """Test that general knowledge queries are routed to web research."""
        planner = RequestPlanner(qmd_url=None, model="gpt-4.1-mini")

        # Test general knowledge queries
        web_queries = [
            "look up the history of indie games",
            "what is the meaning of this symbol?",
            "find out who created this game",
        ]

        for query in web_queries:
            decision = planner.plan(query, None, "")
            assert decision.route == RouteType.WEB_RESEARCH
            assert "web_search" in decision.tools_to_call

    def test_knowledge_base_routing(self):
        """Test that game-specific queries are routed to knowledge base."""
        planner = RequestPlanner(qmd_url=None, model="gpt-4.1-mini")

        # Test with scene context
        scene = {
            "location": "Eastern Forest",
            "activity": "exploring",
            "notable_items": "golden key",
        }

        kb_queries = [
            "what is this item?",
            "where am I?",
            "how do I solve this puzzle?",
        ]

        for query in kb_queries:
            decision = planner.plan(query, scene, "")
            assert decision.route == RouteType.KNOWLEDGE_BASE
            assert "knowledge_base_search" in decision.tools_to_call

    def test_direct_answer_routing(self):
        """Test that simple conversational queries don't trigger retrieval."""
        planner = RequestPlanner(qmd_url=None, model="gpt-4.1-mini")

        # Test simple conversational queries
        direct_queries = [
            "hello",
            "thanks",
            "okay",
            "nice",
        ]

        for query in direct_queries:
            decision = planner.plan(query, None, "")
            assert decision.route == RouteType.DIRECT_ANSWER
            assert len(decision.tools_to_call) == 0

    def test_game_specific_web_routing(self):
        """Test that game-specific queries prefer KB over web."""
        planner = RequestPlanner(qmd_url=None, model="gpt-4.1-mini")

        # Game-specific query should route to KB first, not web
        query = "how do I beat the tunic boss?"
        decision = planner.plan(query, None, "")

        assert decision.route == RouteType.COMBINED
        assert "knowledge_base_search" in decision.tools_to_call
        assert "web_search" not in decision.tools_to_call
