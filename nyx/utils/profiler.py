from dataclasses import dataclass

import jax

BYTES_PER_MB = 1024 * 1024


@dataclass
class MemoryEntry:
    path: str
    shape: tuple
    dtype: str
    bytes: int

    @property
    def mb(self) -> float:
        return self.bytes / BYTES_PER_MB


def _collect_arrays(pytree) -> list[MemoryEntry]:
    """Collect every JAX array leaf in *pytree* with its path."""
    entries = []
    for keypath, leaf in jax.tree_util.tree_leaves_with_path(pytree):
        if isinstance(leaf, jax.Array):
            entries.append(
                MemoryEntry(
                    path=jax.tree_util.keystr(keypath).lstrip(".") or "(root)",
                    shape=tuple(leaf.shape),
                    dtype=str(leaf.dtype),
                    bytes=leaf.size * leaf.dtype.itemsize,
                )
            )
    return entries


def _format_table(entries: list[MemoryEntry], top_n: int = 0) -> str:
    """Format memory entries as a human-readable table."""
    if not entries:
        return "  (no arrays found)\n"

    sorted_entries = sorted(entries, key=lambda e: e.bytes, reverse=True)
    if top_n > 0:
        sorted_entries = sorted_entries[:top_n]

    total = sum(e.bytes for e in entries)
    max_path = max(max(len(e.path) for e in sorted_entries), 4)

    header = f"  {'Path':<{max_path}}  {'Shape':>24}  {'Dtype':>10}  {'MB':>10}  {'%':>6}"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for e in sorted_entries:
        pct = 100 * e.bytes / total if total > 0 else 0
        lines.append(
            f"  {e.path:<{max_path}}  {str(e.shape):>24}  {e.dtype:>10}  "
            f"{e.mb:>10.2f}  {pct:>5.1f}%"
        )
    lines.append("  " + "-" * (len(header) - 2))
    lines.append(f"  {'TOTAL':<{max_path}}  {'':>24}  {'':>10}  {total / BYTES_PER_MB:>10.2f}")
    return "\n".join(lines)


# Public API


def profile_scene(scene, top_n: int = 30) -> dict[str, float]:
    """Profile device memory usage of a Scene pytree.

    Prints a breakdown of memory per component and returns a dict
    mapping component paths to MB.

    Parameters
    ----------
    scene : Scene
        The scene to profile.
    top_n : int
        Show only the top N largest arrays (0 = show all).

    Returns
    -------
    dict[str, float]
        Mapping of path -> megabytes for every array in the scene.
    """
    entries = _collect_arrays(scene)
    total_mb = sum(e.bytes for e in entries) / BYTES_PER_MB

    print(f"\n{'=' * 72}")
    print(f"  Scene memory profile: {len(entries)} arrays, {total_mb:.1f} MB total")
    print(f"{'=' * 72}")

    groups: dict[str, list[MemoryEntry]] = {}
    for e in entries:
        top = e.path.split(".")[0].split("[")[0]
        groups.setdefault(top, []).append(e)

    print("\n  Component summary:")
    for name, group in sorted(groups.items(), key=lambda kv: -sum(e.bytes for e in kv[1])):
        group_mb = sum(e.bytes for e in group) / BYTES_PER_MB
        pct = 100 * group_mb / total_mb if total_mb > 0 else 0
        print(f"    {name:<30}  {group_mb:>8.2f} MB  ({pct:>5.1f}%)")

    print("\n  Top arrays by size:")
    print(_format_table(entries, top_n=top_n))

    _print_device_memory()
    print()

    return {e.path: e.mb for e in entries}


def profile_render(scene, instrument_idx: int = 0) -> dict:
    """Profile device memory during a render pass."""
    backend = jax.default_backend()
    device = jax.devices()[0]
    before = _device_memory_used_mb()
    print(f"\n{'=' * 72}")
    print(f"  Render memory profile (backend={backend}, device={device})")
    print(f"{'=' * 72}")
    print(f"  Memory before render: {before:.1f} MB")

    results = jax.block_until_ready(scene.render())
    after = _device_memory_used_mb()
    peak = _device_peak_memory_mb()

    keys = list(results)
    result = results[keys[instrument_idx]]
    info = {
        "before_mb": before,
        "after_mb": after,
        "peak_mb": peak,
        "render_mb": after - before,
        "result_shape": tuple(result.shape),
    }
    print(f"  Memory after render:  {after:.1f} MB")
    if peak is not None:
        print(f"  Peak memory:          {peak:.1f} MB")
    print(f"  Render overhead:      {after - before:.1f} MB")
    print(f"  Result shape:         {info['result_shape']}")
    for key, r in results.items():
        r_mb = r.size * r.dtype.itemsize / BYTES_PER_MB
        print(f"  Output[{key}]:            {tuple(r.shape)}  {r.dtype}  {r_mb:.2f} MB")
    _print_device_memory()
    print()
    return info


# Device memory helpers


def _device_memory_used_mb() -> float:
    """Current device memory usage in MB (best-effort)."""
    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            return stats.get("bytes_in_use", 0) / BYTES_PER_MB
    except Exception:
        pass
    return 0.0


def _device_peak_memory_mb() -> float | None:
    """Peak device memory usage in MB, or None if unavailable."""
    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            return stats.get("peak_bytes_in_use", 0) / BYTES_PER_MB
    except Exception:
        pass
    return None


def _print_device_memory():
    """Print device memory stats if available."""
    try:
        device = jax.devices()[0]
        stats = device.memory_stats()
        if stats:
            used = stats.get("bytes_in_use", 0) / BYTES_PER_MB
            peak = stats.get("peak_bytes_in_use", 0) / BYTES_PER_MB
            limit = stats.get("bytes_limit", 0) / BYTES_PER_MB
            print(f"\n  Device memory ({device}):")
            print(f"    In use:  {used:>8.1f} MB")
            print(f"    Peak:    {peak:>8.1f} MB")
            if limit > 0:
                print(f"    Limit:   {limit:>8.1f} MB")
                print(f"    Free:    {limit - used:>8.1f} MB")
    except Exception:
        print("\n  (device memory stats not available)")
