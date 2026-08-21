"""Build the local fast-path index (latency-first Tier 1).

READ-ONLY against Neo4j (text export only, no writes) — then locally
re-embeds the corpus with bge-m3 (CUDA) and persists it to
`data/fastpath/`. Run once after the corpus is indexed:

    python -m backend.harness.build_fastpath

Outputs:
  data/fastpath/matrix.npy   float32 (N, 1024) normalized
  data/fastpath/meta.json    chunk_id / text / lang / namespace ...
  data/fastpath/model.txt    model + device provenance
"""

from __future__ import annotations

import asyncio
import logging
import sys

from ..config import get_settings
from ..retrieval.local_index import LocalFastIndex
from ..retrieval.neo4j_store import Neo4jStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> int:
    cfg = get_settings()
    if not cfg.fast_path_namespaces:
        logger.error("fast_path_namespaces is empty — nothing to index")
        return 2
    store = Neo4jStore(cfg)
    try:
        if not await store.verify_connectivity():
            logger.error("Neo4j unreachable at %s", cfg.neo4j_uri)
            return 2
        index = LocalFastIndex(cfg)
        n = await index.build(store)
        logger.info("fast-path index built: %d chunks -> %s", n, cfg.fast_path_index_dir)
        return 0
    finally:
        await store.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))