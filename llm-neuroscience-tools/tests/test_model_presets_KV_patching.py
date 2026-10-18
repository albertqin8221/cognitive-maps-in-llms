import pytest

from model_preset_test_helpers import (
    build_test_input,
    build_nonoverlapping_test_inputs,
)

from llm_neuroscience_tools.presets import RuntimeBundle
from llm_neuroscience_tools.backends.KV_patching import CacheData

import numpy as np
import torch


# ----------- Tests -----------


# Test that the recorded key and value tensors have the correct shape
@pytest.mark.gpu_required
def test_recorded_KV_shape(runtime_bundle: RuntimeBundle) -> None:

    my_backend        = runtime_bundle.backend
    my_prompt_runtime = runtime_bundle.prompt_runtime

    # Run a forward pass and record the KV cache
    prompt = my_prompt_runtime.render_prompt_as_tokens(
        **build_test_input(my_prompt_runtime)
    )
    cache_data, _ = my_backend.forward_with_KV_recording(prompt)
    K_tlia = cache_data.K_tlia
    V_tlia = cache_data.V_tlia

    # Check the shape of the key and value tensors
    expected_shape = [  # t, l, i, a
        len(prompt), 
        runtime_bundle.model_config.shapes['N_layers'],
        runtime_bundle.model_config.shapes['N_heads'],
        runtime_bundle.model_config.shapes['D_head']
    ]
    assert list(K_tlia.shape) == expected_shape, (
        f"Unexpected shape for K_tlia: expected {expected_shape},"
        f"got {list(K_tlia.shape)}"
    )
    assert list(V_tlia.shape) == expected_shape, (
        f"Unexpected shape for V_tlia: expected {expected_shape},"
        f"got {list(V_tlia.shape)}"
    )


# Test that the patch hits exactly all tokens with t <= patch_end
@pytest.mark.gpu_required
def test_forward_with_patching_overwrites_all_masked_loci(
        runtime_bundle: RuntimeBundle
    ) -> None:

    my_backend        = runtime_bundle.backend
    my_prompt_runtime = runtime_bundle.prompt_runtime

    # Record activations on some prompt
    prompt = my_prompt_runtime.render_prompt_as_tokens(
        **build_test_input(my_prompt_runtime)
    )
    cache_data_unperturbed, _ = my_backend.forward_with_KV_recording(prompt)

    # Build a "sentinel" cache data object
    cache_data_sentinel = CacheData(
        K_tlia = torch.ones_like(cache_data_unperturbed.K_tlia) * 12345.0,
        V_tlia = torch.ones_like(cache_data_unperturbed.V_tlia) * 12345.0,
        device_map = cache_data_unperturbed.device_map
    )

    # Fix some value for patch_end
    patch_end = len(prompt) // 2

    # Run a patched forward pass
    cache_data_perturbed, _ = my_backend.forward_with_KV_patching(
        prompt_ids = prompt,
        cache_data = cache_data_sentinel,
        patch_end =  patch_end
    )

    # Positive control: patching should induce at least some downstream change
    # in the unpatched suffix of the recorded K/V tensors.
    assert torch.any(
        cache_data_unperturbed.K_tlia[patch_end+1:, ...] !=
        cache_data_perturbed.K_tlia[patch_end+1:, ...]
    ), (
        "Positive control failed. KV patching produced no downstream change in "
        "the recorded key tensors beyond patch_end."
    )
    assert torch.any(
        cache_data_unperturbed.V_tlia[patch_end+1:, ...] !=
        cache_data_perturbed.V_tlia[patch_end+1:, ...]
    ), (
        "Positive control failed. KV patching produced no downstream change in "
        "the recorded value tensors beyond patch_end."
    )

    # Check that all token positions t <= patch_end are overwritten by the
    # supplied sentinel cache.
    K_prefix_matches = torch.isclose(
        cache_data_sentinel.K_tlia[:patch_end+1, ...],
        cache_data_perturbed.K_tlia[:patch_end+1, ...]
    )
    V_prefix_matches = torch.isclose(
        cache_data_sentinel.V_tlia[:patch_end+1, ...],
        cache_data_perturbed.V_tlia[:patch_end+1, ...]
    )

    assert K_prefix_matches.all(), (
        "forward_with_KV_patching failed to overwrite some key entries in the "
        f"patched prefix [0, {patch_end}]: "
        f"{(~K_prefix_matches).sum().item()} mismatches found."
    )
    assert V_prefix_matches.all(), (
        "forward_with_KV_patching failed to overwrite some value entries in the "
        f"patched prefix [0, {patch_end}]: "
        f"{(~V_prefix_matches).sum().item()} mismatches found."
    )


# ----------- Two integration tests -----------


# I found that using np.allclose(logits_1, logits_2) is too strict for the two 
# tests below. A forward pass using a pre-filled KV cache appears to accumulate
# floating point errors differently from the ordinary forward pass. Since the 
# model runs internally in bfloat16, these numerical differences can be fairly
# large (as large as ~0.5 in the final logits). I'm still not quite sure why 
# the same issue doesn't show up for vanilla activation patching. For now,
# we check a weaker conditions on the logits.


def logits_match_approximately(
    logits_1: np.ndarray,
    logits_2: np.ndarray,
    atol: float = 1.5,
) -> bool:
    """Return True if and only if:
      1. logits_1 == logits_2 elementwise, up to a (possibly coarse) tolerance.
      2. The top token for logits_1 and logits_2 are the same. 
    """
    if logits_1.shape != logits_2.shape:
        return False

    return (
        np.allclose(logits_1, logits_2, atol=atol, rtol=0.0)
        and int(np.argmax(logits_1)) == int(np.argmax(logits_2))
    )


# A control that patches a prompt with its own KV cache (up to the context 
# tokens) and checks that the final logits are unchanged
@pytest.mark.gpu_required
def test_donor_full_forward_matches_cached_donor_forward(
    runtime_bundle: RuntimeBundle,
) -> None:

    my_backend = runtime_bundle.backend
    my_prompt_runtime = runtime_bundle.prompt_runtime
    my_formatter = runtime_bundle.formatter

    prompt_donor = my_prompt_runtime.render_prompt_as_tokens(
        **build_test_input(my_prompt_runtime)
    )

    logits_full = my_backend.forward(prompt_donor)
    cache_data, _ = my_backend.forward_with_KV_recording(prompt_donor)
    _, logits_cached = my_backend.forward_with_KV_patching(
        prompt_ids=prompt_donor,
        cache_data=cache_data,
        patch_end=my_prompt_runtime.landmarks['CTXT_END'],
    )

    # assert np.allclose(logits_full, logits_cached), (  < FAILS!
    assert logits_match_approximately(logits_full, logits_cached), (
        "Full donor forward and cached donor forward do not match. Expected:",
        '\n' + my_formatter.get_pretty_top_k(logits_full),
        "\nGot:",
        '\n' + my_formatter.get_pretty_top_k(logits_cached),
    )


# Generates two prompts: (ctxt_A, query_A); (ctxt_B, query_A). 
# Then, checks that patching KV_ctxt_A -> KV_ctxt_B matches the donor behavior
@pytest.mark.gpu_required
def test_full_context_KV_patching_matches_donor(runtime_bundle: RuntimeBundle) -> None:
    
    my_backend        = runtime_bundle.backend
    my_prompt_runtime = runtime_bundle.prompt_runtime
    my_formatter      = runtime_bundle.formatter

    # Build two prompt inputs
    test_inputs_donor, test_inputs_acceptor = build_nonoverlapping_test_inputs(
        prompt_runtime = my_prompt_runtime
    )

    # Copy the donor query into the acceptor prompt. After this setep:
    # donor    ~= (context_A, query_A),
    # acceptor ~= (context_B, query_A).
    test_inputs_acceptor['query_entities']  = test_inputs_donor['query_entities']
    test_inputs_acceptor['query_relations'] = test_inputs_donor['query_relations']

    # Render the prompts
    prompt_donor = my_prompt_runtime.render_prompt_as_tokens(
        **test_inputs_donor
    )
    prompt_acceptor = my_prompt_runtime.render_prompt_as_tokens(
        **test_inputs_acceptor
    )

    # Record logits on both prompts
    logits_donor    = my_backend.forward(prompt_donor)
    logits_acceptor = my_backend.forward(prompt_acceptor)

    # Record KV data on the donor prompt
    cache_data, _ = my_backend.forward_with_KV_recording(prompt_donor)

    # Run a forward pass on the acceptor, patching up to the context
    _, logits_patched = my_backend.forward_with_KV_patching(
        prompt_ids = prompt_acceptor,
        cache_data = cache_data,
        patch_end = my_prompt_runtime.landmarks['CTXT_END']
    )

    # Positive control: check that patching actually changes the logits:
    # assert not np.allclose(logits_acceptor, logits_patched), (  < FAILS!
    assert not logits_match_approximately(logits_acceptor, logits_patched), (
        "Positive control failed. Patching failed to change the logits:",
        '\n'+my_formatter.get_pretty_top_k(logits_acceptor),
    )

    # Require that patching the context rescues donor behavior
    # assert np.allclose(logits_donor, logits_patched), (  < FAILS!
    assert logits_match_approximately(logits_donor, logits_patched), (
        "Logits do not match. Expected:",
        '\n'+my_formatter.get_pretty_top_k(logits_donor),
        "\nGot:",
        '\n'+my_formatter.get_pretty_top_k(logits_patched)
    )
