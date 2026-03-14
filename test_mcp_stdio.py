import logging
import sys
import os
import json

# Add src to path if needed
sys.path.append(os.path.abspath("src"))

from rag.qmd_client import QmdMcpStdioClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_test")


def test_list_tools():
    """Test listing available MCP tools."""
    client = QmdMcpStdioClient(index_name="index")
    client._start()

    logger.info("Listing available MCP tools...")
    result = client._call("tools/list", {})
    logger.info(f"Available tools: {json.dumps(result, indent=2)}")

    if client._process:
        client._process.terminate()


def test_mcp_query():
    # Use the 'tunic' collection found in 'qmd collection list'
    # The default index name is 'index', but qmd status showed it's the one being used
    client = QmdMcpStdioClient(index_name="index")

    query_text = "What is Tunic?"
    collection = "tunic"

    logger.info(f"Running MCP query: '{query_text}' in collection '{collection}'")

    try:
        results = client.query(query_text, collection, limit=3)

        if not results:
            logger.warning("No results found.")
        else:
            logger.info(f"Found {len(results)} results:")
            for i, res in enumerate(results):
                print(f"\n--- Result {i + 1} (Score: {res.score:.2f}) ---")
                print(f"File: {res.file}")
                print(f"DocID: {res.docid}")
                print(f"Snippet: {res.content[:200]}...")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        # Cleanup is handled by __del__ but we can be explicit if we added a close()
        if hasattr(client, "_process") and client._process:
            logger.info("Terminating MCP process...")
            client._process.terminate()


if __name__ == "__main__":
    test_list_tools()
    print("\n" + "=" * 80 + "\n")
    test_mcp_query()
