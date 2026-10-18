from __future__ import annotations

import torch
import einops

from transformers.cache_utils import DynamicCache


class CacheData:

    def __init__(self, 
                 K_tlia: torch.Tensor, 
                 V_tlia: torch.Tensor,
                 device_map: list[tuple[torch.device, torch.device]]
                 ) -> None:
        """Build a CacheData object from unbatched key and value tensors.

        The expected shape is ``[T_context, N_layers, N_heads, D_head]``.
        """
        assert K_tlia.shape == V_tlia.shape, (
            "K_tlia and V_tlia must have the same shape; "
            f"got K_tlia.shape={tuple(K_tlia.shape)} and "
            f"V_tlia.shape={tuple(V_tlia.shape)}"
        )
        self.K_tlia     = K_tlia
        self.V_tlia     = V_tlia
        self.device_map = device_map
        

    def empty_like(self) -> CacheData:
        """Creates an empty CacheData object with matching shape.
        """
        return CacheData(
            torch.zeros_like(self.K_tlia), 
            torch.zeros_like(self.V_tlia),
            self.device_map
        )
    
    def clone(self) -> CacheData:
        """Clones a CacheData object.
        """
        return CacheData(
            self.K_tlia.clone(), 
            self.V_tlia.clone(),
            self.device_map
        )
    
    @classmethod
    def from_dynamic_cache(cls, 
                           dynamic_cache: DynamicCache,
                           storage_device: torch.device | str
                           ) -> CacheData:
        """Build a CacheData object from a Hugging Face DynamicCache.

        Args:
            dynamic_cache: A Hugging Face ``DynamicCache`` from a 
            batch-1 forward pass
            storage_device: Device on which to store the ``K_tlia`` and 
            ``V_tlia`` tensors in the returned ``CacheData``.

        Returns:
            A new ``CacheData`` object whose ``K_tlia`` and ``V_tlia`` tensors
            have shape ``[T_context, N_layers, N_heads, D_head]`` and whose
            ``device_map`` records the original key/value device for each layer.
        """

        # Extract some data from the DynamicCache
        N_layers = len(dynamic_cache.layers)
        example_keys = dynamic_cache.layers[0].keys
        assert example_keys is not None
        bs, N_heads, T_context, D_head = example_keys.shape
        assert bs == 1, f"Expected batch size 1 in DynamicCache; got {bs}"
        dtype  = example_keys.dtype

        # Preallocate tensors to store keys and values
        K_tlia = torch.zeros((T_context, N_layers, N_heads, D_head),
                        device=storage_device, dtype=dtype)
        V_tlia = torch.zeros((T_context, N_layers, N_heads, D_head),
                        device=storage_device, dtype=dtype)
        device_map = []

        # Populate the tensors
        for l in range(len(dynamic_cache.layers)):
            layer = dynamic_cache.layers[l]
            assert layer.keys is not None
            assert layer.values is not None
            K_tia = einops.rearrange(layer.keys[0, ...],   'i t a -> t i a')
            V_tia = einops.rearrange(layer.values[0, ...], 'i t a -> t i a')
            K_tlia[:, l, :, :] = K_tia.to(storage_device)
            V_tlia[:, l, :, :] = V_tia.to(storage_device)
            device_map.append((layer.keys.device, layer.values.device))

        return CacheData(K_tlia, V_tlia, device_map)

    def to_dynamic_cache(self, max_token_idx: int = -1) -> DynamicCache:
        """Convert this `CacheData` object to a new `DynamicCache`.

        Args:
            max_token_idx: Inclusive upper bound on the token positions to include.
                If set to `-1`, include the full token range.

        Returns:
            A new `DynamicCache` containing KV entries up to and including
            `max_token_idx`.
        """
        T_context = self.K_tlia.shape[0]
        if max_token_idx == -1:
            max_token_idx = T_context-1
        assert 0 <= max_token_idx and max_token_idx < T_context, (
            "max_token_idx must satisfy 0 <= max_token_idx < T_context; "
            f"got T_context={T_context} and "
            f"max_token_idx={max_token_idx}"
        )

        # Populate the DynamicCache object
        cache = DynamicCache()
        for l in range(self.K_tlia.shape[1]):
            K_btia = einops.rearrange(
                self.K_tlia[:, l, :, :],
                "t i a -> i t a",
            )[None, :, : max_token_idx + 1, :]
            V_btia = einops.rearrange(
                self.V_tlia[:, l, :, :],
                "t i a -> i t a",
            )[None, :, : max_token_idx + 1, :]
            K_btia = K_btia.to(self.device_map[l][0])
            V_btia = V_btia.to(self.device_map[l][1])
            cache.update(K_btia, V_btia, l)

        return cache
    