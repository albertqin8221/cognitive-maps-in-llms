import pytest
import torch

from llm_neuroscience_tools.backends.KV_patching import CacheData


def make_sample_cache_data() -> CacheData:
    shape = (4, 2, 3, 5)
    K_tlia = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
    K_tlia = K_tlia.reshape(shape)
    V_tlia = (1000 + K_tlia).clone()
    device_map = [
        (torch.device("cpu"), torch.device("cpu"))
        for _ in range(shape[1])
    ]
    return CacheData(K_tlia=K_tlia, V_tlia=V_tlia, device_map=device_map)


# Test that CacheData enforces matching key/value tensor shapes
def test_init_rejects_mismatched_shapes() -> None:
    K_tlia = torch.zeros((4, 2, 3, 5))
    V_tlia = torch.zeros((5, 2, 3, 5))
    device_map = [
        (torch.device("cpu"), torch.device("cpu"))
        for _ in range(K_tlia.shape[1])
    ]

    with pytest.raises(AssertionError, match="must have the same shape"):
        CacheData(K_tlia=K_tlia, V_tlia=V_tlia, device_map=device_map)


# Test that clone preserves values but allocates independent storage
def test_clone_returns_equal_values_with_independent_storage() -> None:
    cache_data = make_sample_cache_data()
    cache_data_clone = cache_data.clone()

    assert torch.equal(cache_data_clone.K_tlia, cache_data.K_tlia)
    assert torch.equal(cache_data_clone.V_tlia, cache_data.V_tlia)
    assert cache_data_clone.device_map == cache_data.device_map

    cache_data_clone.K_tlia[0, 0, 0, 0] = -1
    cache_data_clone.V_tlia[0, 0, 0, 0] = -2

    assert cache_data.K_tlia[0, 0, 0, 0].item() != -1
    assert cache_data.V_tlia[0, 0, 0, 0].item() != -2


# Test that empty_like preserves metadata and zero-initializes both tensors
def test_empty_like_preserves_metadata_and_is_zero_filled() -> None:
    cache_data = make_sample_cache_data()
    empty_cache_data = cache_data.empty_like()

    assert empty_cache_data.K_tlia.shape == cache_data.K_tlia.shape
    assert empty_cache_data.V_tlia.shape == cache_data.V_tlia.shape
    assert empty_cache_data.K_tlia.dtype == cache_data.K_tlia.dtype
    assert empty_cache_data.V_tlia.dtype == cache_data.V_tlia.dtype
    assert empty_cache_data.K_tlia.device == cache_data.K_tlia.device
    assert empty_cache_data.V_tlia.device == cache_data.V_tlia.device
    assert empty_cache_data.device_map == cache_data.device_map
    assert torch.count_nonzero(empty_cache_data.K_tlia) == 0
    assert torch.count_nonzero(empty_cache_data.V_tlia) == 0

    empty_cache_data.K_tlia[0, 0, 0, 0] = 7
    empty_cache_data.V_tlia[0, 0, 0, 0] = 8

    assert cache_data.K_tlia[0, 0, 0, 0].item() != 7
    assert cache_data.V_tlia[0, 0, 0, 0].item() != 8


# Test that to_dynamic_cache returns a cache truncated to the requested token prefix
def test_to_dynamic_cache_truncates_to_exact_prefix_length() -> None:
    cache_data = make_sample_cache_data()
    max_token_idx = 2

    dynamic_cache = cache_data.to_dynamic_cache(max_token_idx=max_token_idx)

    for layer_idx in range(len(dynamic_cache.layers)):
        layer = dynamic_cache.layers[layer_idx]
        assert layer.keys is not None
        assert layer.values is not None
        assert layer.keys.shape[2] == max_token_idx + 1
        assert layer.values.shape[2] == max_token_idx + 1


# A "round trip test". We start with CacheData, convert/truncate it to a 
# DynamicCache, then convert it back to CacheData, and check the data is the same
def test_dynamic_cache_roundtrip_preserves_truncated_prefix() -> None:
    cache_data = make_sample_cache_data()
    max_token_idx = 2

    dynamic_cache = cache_data.to_dynamic_cache(max_token_idx=max_token_idx)
    recovered_cache_data = CacheData.from_dynamic_cache(
        dynamic_cache,
        storage_device="cpu",
    )

    assert torch.equal(
        recovered_cache_data.K_tlia,
        cache_data.K_tlia[: max_token_idx + 1, :, :, :],
    )
    assert torch.equal(
        recovered_cache_data.V_tlia,
        cache_data.V_tlia[: max_token_idx + 1, :, :, :],
    )
    assert recovered_cache_data.device_map == cache_data.device_map
