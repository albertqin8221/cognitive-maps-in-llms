import pytest

import torch
import numpy as np

from llm_neuroscience_tools.presets import (
    RuntimeBundle
)
from model_preset_test_helpers import (
    build_nonoverlapping_test_inputs,
    build_test_input
)


# ----------- Tests -----------


# Smoke test that runs a forward pass and prints for inspection
@pytest.mark.gpu_required
def test_forward_pass_smoke(runtime_bundle: RuntimeBundle) -> None:

    my_backend        = runtime_bundle.backend
    my_prompt_runtime = runtime_bundle.prompt_runtime
    my_formatter      = runtime_bundle.formatter

    prompt = my_prompt_runtime.render_prompt_as_tokens(
        **build_test_input(my_prompt_runtime)
    )
    logits = my_backend.forward(prompt)

    print()
    print(f'Smoke test for backend `{runtime_bundle.model_config.name}`',
          f'using `{runtime_bundle.prompt_config.name}` prompt family:')
    print(my_formatter.get_prompt_text(prompt))
    print(my_formatter.get_pretty_top_k(logits))


# Test that the recorded activation tensor has the correct shape
@pytest.mark.gpu_required
def test_recorded_shape(runtime_bundle: RuntimeBundle) -> None:

    my_backend        = runtime_bundle.backend
    my_prompt_runtime = runtime_bundle.prompt_runtime

    prompt = my_prompt_runtime.render_prompt_as_tokens(
        **build_test_input(my_prompt_runtime)
    )
    Z_tla, _ = my_backend.forward_with_recording(prompt)

    expected_shape = [
        len(prompt), 
        runtime_bundle.model_config.shapes['N_layers'],
        runtime_bundle.model_config.shapes['D_model']
    ]

    assert list(Z_tla.shape) == expected_shape, (
        f"Unexpected shape for Z_tla: expected {expected_shape},"
        f"got {list(Z_tla.shape)}"
    )


# Test that the patch hits exactly all loci where mask_tla = 1
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
    Z_tla_unperturbed, _ = my_backend.forward_with_recording(prompt)

    # Build a "sentinel" activation tensor
    Z_tla_sentinel = torch.ones_like(Z_tla_unperturbed) * 12345.0

    # Build a random patching mask
    g = torch.Generator(device=Z_tla_unperturbed.device).manual_seed(0)
    mask_tla = torch.rand(
        Z_tla_unperturbed.shape, 
        device=Z_tla_unperturbed.device, 
        generator=g
    ) < 0.5

    # Run a patched forward pass
    Z_tla_perturbed, _ = my_backend.forward_with_patching(
        prompt_ids = prompt,
        Z_tla = Z_tla_sentinel,
        mask_tla = mask_tla
    )

    # Positive control: patching should induce at least some downstream change
    assert torch.any(
        Z_tla_unperturbed[mask_tla==0] != 
        Z_tla_perturbed[mask_tla==0]
    ), (
        "Positive control failed. Activation patching produced no change "
        "outside the directly masked entries."
    )

    # Check that the patch hits all loci where mask_tla = 1
    mismatch_mask = ~torch.isclose(
        Z_tla_sentinel[mask_tla],
        Z_tla_perturbed[mask_tla]
    )
    assert not mismatch_mask.any(), (
        "forward_with_patching failed to overwrite some activations: "
        f"{mismatch_mask.sum().item()} / {mismatch_mask.numel()} masked entries "
        "did not match the supplied patch values."
    )


# ----------- Two integration tests -----------


# A control that patches a prompt with its own context activations and checks 
# that the final logits are unchanged
@pytest.mark.gpu_required
def test_noop_full_context_patching_preserves_logits(
        runtime_bundle: RuntimeBundle
    ) -> None:

    my_backend        = runtime_bundle.backend
    my_prompt_runtime = runtime_bundle.prompt_runtime
    my_formatter      = runtime_bundle.formatter

    # Build a single prompt
    prompt = my_prompt_runtime.render_prompt_as_tokens(
        **build_test_input(my_prompt_runtime)
    )

    # Record logits and activations on the same prompt
    logits_unperturbed = my_backend.forward(prompt)
    Z_tla, _ = my_backend.forward_with_recording(prompt)

    # Construct a patching mask to patch at context tokens only
    CTXT_START = my_prompt_runtime.landmarks['CTXT_START']
    CTXT_END   = my_prompt_runtime.landmarks['CTXT_END']
    mask_tla = torch.zeros_like(Z_tla, dtype=torch.bool)
    mask_tla[CTXT_START:CTXT_END+1, ...] = True

    # Run a forward pass on the same prompt, patching in the same context activations
    _, logits_patched = my_backend.forward_with_patching(prompt, Z_tla, mask_tla)

    assert np.allclose(logits_unperturbed, logits_patched), (
        "No-op context patching changed the logits. Expected:",
        '\n'+my_formatter.get_pretty_top_k(logits_unperturbed),
        "\nGot:",
        '\n'+my_formatter.get_pretty_top_k(logits_patched)
    )


# Generates two prompts: (ctxt_A, query_A); (ctxt_B, query_A). 
# Then, checks that patching ctxt_A -> ctxt_B matches the donor behavior
@pytest.mark.gpu_required
def test_full_context_patching_matches_donor(runtime_bundle: RuntimeBundle) -> None:
    
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

    # Record activations on the donor prompt
    Z_tla, _ = my_backend.forward_with_recording(prompt_donor)

    # Construct a patching mask to patch at context tokens only
    CTXT_START = my_prompt_runtime.landmarks['CTXT_START']
    CTXT_END   = my_prompt_runtime.landmarks['CTXT_END']
    mask_tla = torch.zeros_like(Z_tla, dtype=torch.bool)
    mask_tla[CTXT_START:CTXT_END+1, ...] = True

    # Run a forward pass on the acceptor, patching at all context tokens
    _, logits_patched = my_backend.forward_with_patching(prompt_acceptor, Z_tla, mask_tla)

    # Positive control: check that patching actually changes the logits:
    assert not np.allclose(logits_acceptor, logits_patched), (
        "Positive control failed. Patching failed to change the logits:",
        '\n'+my_formatter.get_pretty_top_k(logits_acceptor),
    )

    # Require that patching the context rescues donor behavior
    assert np.allclose(logits_donor, logits_patched), (
        "Logits do not match. Expected:",
        '\n'+my_formatter.get_pretty_top_k(logits_donor),
        "\nGot:",
        '\n'+my_formatter.get_pretty_top_k(logits_patched)
    )


