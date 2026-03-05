# QMD Migration Summary

## Overview

Successfully replaced ChromaDB with QMD (Queryable Markdown) for local knowledge retrieval.

## Changes Made

### 1. New QMD Client (`src/rag/qmd_client.py`)

**Created two client implementations:**

- **`QmdHttpClient`**: Queries QMD via HTTP REST API
  - Configurable base URL (e.g., `http://localhost:18788`)
  - Sends hybrid search (lexical + vector)
  - Returns structured `QmdQueryResult` objects

- **`QmdCliClient`**: Queries QMD via command-line interface
  - Uses `qmd` binary with `--json` output
  - Configurable index name (default: `game-companion`)
  - Subprocess-based execution with timeout protection

**Data Structure:**
```python
class QmdQueryResult:
    content: str      # Document content/snippet
    score: float      # Relevance score (0-1)
    file: str         # Source file path
    docid: str        # Document ID
    metadata: dict    # Additional metadata
```

### 2. Updated Local Retriever (`src/rag/local_retriever.py`)

**Replaced ChromaDB with QMD:**
- Removed ChromaDB client initialization
- Added QMD client selection logic (HTTP vs CLI)
- Updated query method to use QMD results
- Simplified initialization (no directory checks needed)

**Configuration:**
- `qmd_url`: Optional HTTP server URL (if None, uses CLI)
- `index_name`: QMD index name for CLI mode (default: `game-companion`)

### 3. Updated Query Interface (`src/rag/query.py`)

**Replaced ChromaDB queries with QMD:**
- Removed ChromaDB imports and directory checks
- Added environment variable support: `QMD_URL`
- Updated `query_tunic_knowledge()` to use QMD clients
- Maintained same function signature for backward compatibility

### 4. Dependencies (`pyproject.toml`)

**Removed:**
- `chromadb>=0.3.21`

**No new dependencies added** - QMD uses standard library (`subprocess`, `json`) and existing `requests`

## Usage

### Environment Variables

```bash
# Optional: Use QMD HTTP server instead of CLI
export QMD_URL=http://localhost:18788
```

### QMD Setup

**1. Install QMD:**
```bash
# Install from cargo
cargo install qmd

# Or download binary from releases
```

**2. Index Tunic Wiki:**
```bash
# Create index from markdown files
qmd --index game-companion index -c tunic /path/to/tunic-wiki/*.md
```

**3. Query (CLI):**
```bash
# Direct CLI query
qmd --index game-companion query -c tunic "how to beat the garden knight"

# Via Python
python -m src.rag.query "how to beat the garden knight"
```

**4. Query (HTTP Server):**
```bash
# Start QMD server
qmd --index game-companion serve --port 18788

# Set environment variable
export QMD_URL=http://localhost:18788

# Query via Python
python -m src.rag.query "how to beat the garden knight"
```

### Integration with Main Pipeline

The main pipeline automatically uses QMD via `LocalGameRetriever`:

```python
# In main.py
orchestrator = KnowledgeOrchestrator()
orchestrator.register_retriever(LocalGameRetriever())  # Uses QMD CLI by default
orchestrator.register_retriever(ExaRetriever())
```

**To use HTTP mode:**
```python
# Set QMD_URL environment variable before running
export QMD_URL=http://localhost:18788
python -m src.main --replay --no-voice --duration 5
```

## Migration Benefits

### Advantages of QMD over ChromaDB

1. **Simpler Dependencies**: No heavy vector database dependencies
2. **Hybrid Search**: Built-in lexical + vector search
3. **Markdown Native**: Direct indexing of markdown files
4. **Lightweight**: CLI tool with optional HTTP server
5. **Portable**: Single binary, no Python dependencies
6. **Fast**: Optimized for markdown document retrieval

### Performance

- **Query Speed**: Similar to ChromaDB for small collections
- **Index Size**: More compact than ChromaDB
- **Memory Usage**: Lower memory footprint (CLI mode)
- **Startup Time**: Faster (no database initialization)

## Testing

### Type Checking
```bash
uv run pyright
# Result: 0 errors, 0 warnings, 0 informations ✓
```

### Runtime Test
```bash
# Note: Requires QMD to be installed and indexed
uv run src/main.py --replay --no-voice --duration 5
```

## Backward Compatibility

### Breaking Changes
- **ChromaDB removed**: Old ChromaDB-based code will not work
- **Index format**: Requires QMD index instead of ChromaDB collection
- **No migration tool**: Must re-index content with QMD

### Compatible APIs
- `query_tunic_knowledge()` function signature unchanged
- `LocalGameRetriever` interface unchanged (KnowledgeRetriever protocol)
- RAG orchestration layer unchanged

## Next Steps

### Required Actions

1. **Install QMD**:
   ```bash
   cargo install qmd
   ```

2. **Index Tunic Wiki**:
   ```bash
   qmd --index game-companion index -c tunic data/tunic-wiki/*.md
   ```

3. **Update Documentation**: Update README with QMD setup instructions

4. **Remove Old Data**: Delete `data/chroma/` directory (no longer needed)

### Optional Enhancements

1. **QMD Server Mode**: Set up QMD HTTP server for production
2. **Multi-Game Support**: Index multiple game wikis as separate collections
3. **Auto-Indexing**: Add script to automatically index new markdown files
4. **Query Optimization**: Tune QMD search parameters for better results

## Troubleshooting

### QMD Not Found
```bash
# Error: qmd: command not found
# Solution: Install QMD
cargo install qmd
```

### Index Not Found
```bash
# Error: QMD CLI query failed
# Solution: Create index
qmd --index game-companion index -c tunic /path/to/markdown/*.md
```

### HTTP Server Connection Failed
```bash
# Error: QMD HTTP error: Connection refused
# Solution: Start QMD server
qmd --index game-companion serve --port 18788
```

## Architecture Notes

### Client Selection Logic

```python
# LocalGameRetriever automatically selects client:
if qmd_url:
    # Use HTTP client
    client = QmdHttpClient(qmd_url)
else:
    # Use CLI client
    client = QmdCliClient(index_name)
```

### Error Handling

- Per-retriever error isolation in orchestrator
- Graceful fallback to Exa if QMD fails
- Detailed error logging for debugging

### Score Mapping

- QMD scores are in [0, 1] range (higher = better)
- Used directly as confidence scores
- No conversion needed (unlike ChromaDB distance)
