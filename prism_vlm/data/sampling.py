"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Deterministic sampling of ``items_per_benchmark`` items per benchmark, per seed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..config.loader import DomainsConfig
from ..utils.io import write_json_atomic
from .registry import SourceNotProvidedError, get_loader
from .schemas import Item

_ITEMS_META_FILE = "items_meta.jsonl"


def deterministic_sample_indices(pool_size: int, k: int, seed: int) -> list[int]:
    """Pick ``k`` indices without replacement from ``range(pool_size)`` under ``seed``."""
    if pool_size < k:
        raise ValueError(f"Pool too small ({pool_size}) for k={k}")
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(pool_size, size=k, replace=False))


def _serialise_item(item: Item) -> dict:
    """JSON-safe dict for ``Item`` minus images (too large to persist)."""
    gold = list(item.gold) if isinstance(item.gold, tuple) else item.gold
    return {
        "item_id": item.item_id,
        "domain": item.domain,
        "benchmark": item.benchmark,
        "question": item.question,
        "gold": gold,
        "judge_gold": item.judge_gold,
        "meta": item.meta,
    }


def _deserialise_item(d: dict) -> Item:
    gold = tuple(d["gold"]) if isinstance(d["gold"], list) else d["gold"]
    return Item(
        item_id=d["item_id"],
        domain=d["domain"],
        benchmark=d["benchmark"],
        question=d["question"],
        images=(),  # hydrate separately when needed
        gold=gold,
        judge_gold=d.get("judge_gold"),
        meta=d.get("meta", {}) or {},
    )


def _write_items_metadata(base_dir: Path, items_by_id: dict[str, Item]) -> None:
    import os
    path = base_dir / _ITEMS_META_FILE
    # PID suffix avoids collisions when two parallel processes both land in
    # the full-load branch (deterministic content → last rename wins).
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        for item in items_by_id.values():
            fh.write(json.dumps(_serialise_item(item), ensure_ascii=False) + "\n")
    tmp.replace(path)


def _items_metadata_to_write(
    base_dir: Path, item_by_id: dict[str, Item], is_partial: bool
) -> dict[str, Item]:
    """What ``items_meta.jsonl`` should hold after this call.

    A filtered call saw only part of the pool, so it merges with what is
    already on disk, exactly as the two id maps above do. Overwriting instead
    would drop every item outside the filter, and the ``load_images=False``
    fast path would then silently return a shrunken pool on the next
    unfiltered run.
    """
    if not is_partial:
        return item_by_id
    return {**_read_items_metadata(base_dir), **item_by_id}


def _read_items_metadata(base_dir: Path) -> dict[str, Item]:
    path = base_dir / _ITEMS_META_FILE
    if not path.exists():
        return {}
    out: dict[str, Item] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out[json.loads(line)["item_id"]] = _deserialise_item(json.loads(line))
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def _image_cache_path(cache_dir: Path, bench: str, item_id: str) -> Path:
    return cache_dir / ".images" / bench / f"{item_id}.imgs"


def _read_image_cache(cache_dir: Path, bench: str, item_id: str) -> tuple[bytes, ...] | None:
    """Read the per-item image-bytes cache. Returns None on miss / corrupt file.

    Format: 4-byte LE count, then for each image: 4-byte LE length + bytes.
    """
    p = _image_cache_path(cache_dir, bench, item_id)
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    try:
        n = int.from_bytes(raw[:4], "little")
        out: list[bytes] = []
        off = 4
        for _ in range(n):
            ln = int.from_bytes(raw[off:off+4], "little")
            off += 4
            out.append(raw[off:off+ln])
            off += ln
        return tuple(out)
    except (IndexError, ValueError):
        return None


def _write_image_cache(cache_dir: Path, bench: str, item_id: str, images: tuple[bytes, ...]) -> None:
    p = _image_cache_path(cache_dir, bench, item_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    buf = bytearray()
    buf += len(images).to_bytes(4, "little")
    for img in images:
        buf += len(img).to_bytes(4, "little")
        buf += img
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(bytes(buf))
    tmp.replace(p)


def hydrate_items(
    items: list[Item],
    domains_cfg: DomainsConfig,
    image_cache_dir: Path | None = None,
) -> list[Item]:
    """Re-fetch images for any items whose ``images`` tuple is empty.

    With ``image_cache_dir`` set, image bytes are cached on disk per item under
    ``<image_cache_dir>/.images/<bench>/<item_id>.imgs`` so subsequent runs of
    any model can skip the expensive HF dataset open + image decode step.
    Cache hits read raw bytes directly from disk; misses fall back to the HF
    loader and write the cache afterwards.
    """
    pending: list[Item] = []
    hydrated_by_id: dict[str, Item] = {}
    for item in items:
        if item.images:
            continue
        if image_cache_dir is not None:
            cached = _read_image_cache(image_cache_dir, item.benchmark, item.item_id)
            if cached:
                hydrated_by_id[item.item_id] = Item(
                    item_id=item.item_id,
                    domain=item.domain,
                    benchmark=item.benchmark,
                    question=item.question,
                    images=cached,
                    gold=item.gold,
                    judge_gold=item.judge_gold,
                    meta=item.meta,
                )
                continue
        pending.append(item)

    if pending:
        by_bench: dict[str, list[Item]] = {}
        for item in pending:
            by_bench.setdefault(item.benchmark, []).append(item)
        bench_spec = {b.id: b for b in domains_cfg.all_benchmarks}
        for bench_id, bench_items in by_bench.items():
            spec = bench_spec.get(bench_id)
            if spec is None:
                continue
            loader = get_loader(spec)
            for it in bench_items:
                orig_idx = it.meta.get("orig_index")
                if orig_idx is None:
                    continue
                fetched = loader.item_at(it.domain, int(orig_idx))
                hydrated_by_id[it.item_id] = fetched
                if image_cache_dir is not None and fetched.images:
                    try:
                        _write_image_cache(image_cache_dir, it.benchmark, it.item_id, fetched.images)
                    except OSError:
                        pass

    return [hydrated_by_id.get(it.item_id, it) for it in items]


def take_sample(items: list[Item], limit: int | None) -> list[Item]:
    """Deterministic subset for debug runs.

    Sorted by item id first: the sampler returns the union in set order, which
    CPython varies per process, so slicing it directly would hand build and run
    two different subsets.
    """
    if limit is None or limit >= len(items):
        return items
    return sorted(items, key=lambda it: it.item_id)[:limit]


def load_sampled_items(
    domains_cfg: DomainsConfig,
    base_dir: Path,
    seeds: Iterable[int],
    only_benchmarks: Iterable[str] | None = None,
    only_domains: Iterable[str] | None = None,
    *,
    load_images: bool = True,
) -> tuple[list[Item], dict[int, list[str]]]:
    """Sample K items per benchmark for every seed in ``seeds``; return the union.

    K is ``sampling.items_per_benchmark``. Each seed picks its own K indices
    per benchmark, persisted to ``<base>/<bench>/seed_<N>/indices.json``. Item ids are content-keyed
    (``sha1(benchmark|orig_index|question)[:16]``) so overlapping picks share
    the same cache entry — the returned ``items`` is the union (no duplicates).

    Also writes three sidecar files under ``<base>/``:
      - ``item_domain_map.json``: ``{item_id: domain}`` for monitor bucketing.
      - ``seed_items_map.json``: ``{str(seed): [item_id, ...]}``, which seed
        drew which items. ``prism-vlm score`` aggregates the pool flat, as the
        published numbers do; this is here for per-seed analyses.
      - ``items_meta.jsonl``: per-item metadata (id, question, gold, meta)
        **without images**. Lets subsequent runs skip HF dataset loading;
        images are re-fetched via :func:`hydrate_items` only for items whose
        inference cache is missing.

    ``load_images=False`` is a fast path: when the metadata file already
    exists and covers the requested seeds, skip HF dataset access entirely
    and return ``Item``\\ s with empty image tuples. The caller is expected
    to call :func:`hydrate_items` on whichever subset still needs inference.

    Returns ``(items, seed_to_item_ids)``.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    k = domains_cfg.sampling.items_per_benchmark
    seeds_list = list(seeds)

    bench_filter = set(only_benchmarks) if only_benchmarks else None
    dom_filter = set(only_domains) if only_domains else None

    # Fast path: metadata + per-seed indices already persisted → skip HF.
    if not load_images and not bench_filter and not dom_filter:
        seed_map_path = base_dir / "seed_items_map.json"
        meta_path = base_dir / _ITEMS_META_FILE
        if seed_map_path.exists() and meta_path.exists():
            try:
                raw = json.loads(seed_map_path.read_text())
                cached_seeds: dict[int, list[str]] = {
                    int(s): list(v) for s, v in raw.items()
                }
                if set(seeds_list).issubset(cached_seeds.keys()):
                    items_by_id = _read_items_metadata(base_dir)
                    if items_by_id:
                        seed_to_ids = {s: cached_seeds[s] for s in seeds_list}
                        referenced = {i for ids in seed_to_ids.values() for i in ids}
                        items = [items_by_id[i] for i in referenced if i in items_by_id]
                        return items, seed_to_ids
            except (OSError, json.JSONDecodeError):
                pass  # fall through to full load

    item_by_id: dict[str, Item] = {}
    seed_to_item_ids: dict[int, list[str]] = {s: [] for s in seeds_list}
    item_domain_map: dict[str, str] = {}

    for domain, benches in domains_cfg.domains.items():
        if dom_filter and domain not in dom_filter:
            continue
        for bench in benches:
            if bench_filter and bench.id not in bench_filter:
                continue
            try:
                loader = get_loader(bench)
            except SourceNotProvidedError as e:
                print(f"[sample] skipping {bench.id}: {e}", flush=True)
                continue
            pool_size: int | None = None
            for seed in seeds_list:
                idx_path = base_dir / bench.id / f"seed_{seed}" / "indices.json"
                if idx_path.exists():
                    with idx_path.open() as fh:
                        indices = json.load(fh)
                else:
                    if pool_size is None:
                        pool_size = loader.pool_size()
                    indices = deterministic_sample_indices(pool_size, k, seed)
                    write_json_atomic(idx_path, indices)
                for idx in indices:
                    item = loader.item_at(domain, idx)
                    item_by_id[item.item_id] = item
                    item_domain_map[item.item_id] = domain
                    seed_to_item_ids[seed].append(item.item_id)

    # Sidecar maps for monitor + scoring. When only_{benchmarks,domains} is set
    # we do a partial overwrite (merge) so prior seed/domain bundles survive.
    is_partial = bool(bench_filter or dom_filter)

    dom_map_path = base_dir / "item_domain_map.json"
    if is_partial and dom_map_path.exists():
        try:
            existing = json.loads(dom_map_path.read_text())
            existing.update(item_domain_map)
            item_domain_map = existing
        except (OSError, json.JSONDecodeError):
            pass
    write_json_atomic(dom_map_path, item_domain_map)

    seed_map_path = base_dir / "seed_items_map.json"
    seed_map_to_write = {str(s): ids for s, ids in seed_to_item_ids.items()}
    if is_partial and seed_map_path.exists():
        try:
            existing = json.loads(seed_map_path.read_text())
            for k_s, v in seed_map_to_write.items():
                existing.setdefault(k_s, [])
                merged = set(existing[k_s]) | set(v)
                existing[k_s] = sorted(merged)
            seed_map_to_write = existing
        except (OSError, json.JSONDecodeError):
            pass
    write_json_atomic(seed_map_path, seed_map_to_write)

    # items_meta.jsonl: store everything except images, so future runs can
    # skip HF dataset loading via the ``load_images=False`` fast path.
    _write_items_metadata(
        base_dir, _items_metadata_to_write(base_dir, item_by_id, is_partial)
    )

    return list(item_by_id.values()), seed_to_item_ids
