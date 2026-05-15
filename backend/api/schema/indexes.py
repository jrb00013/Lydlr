"""Ensure MongoDB indexes at application startup."""
import logging

from backend.api.schema.collections import COLLECTIONS, INDEX_SPECS, METRICS_TTL_SECONDS

logger = logging.getLogger(__name__)


async def ensure_indexes(db) -> None:
    if db is None:
        return

    for collection_name, keys, options in INDEX_SPECS:
        try:
            await db[collection_name].create_index(keys, **options)
        except Exception as exc:
            logger.warning("Index %s on %s: %s", keys, collection_name, exc)

    # TTL on raw metrics (idempotent — ignore if exists with different opts)
    try:
        await db[COLLECTIONS["METRICS"]].create_index(
            [("timestamp", 1)],
            expireAfterSeconds=METRICS_TTL_SECONDS,
            name="metrics_ttl_7d",
        )
    except Exception as exc:
        logger.debug("Metrics TTL index: %s", exc)

    logger.info("MongoDB indexes ensured for Lydlr collections")
