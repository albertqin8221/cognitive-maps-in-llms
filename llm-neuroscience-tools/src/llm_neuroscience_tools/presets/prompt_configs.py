"""Prompt preset definitions."""

from __future__ import annotations

from dataclasses import dataclass

from llm_neuroscience_tools.prompts.relational import RelationalPromptSchema, RelationPool


@dataclass(frozen=True)
class PromptConfig:
    """Prompt configuration used to build runtimes."""

    name: str
    prompt_schema: RelationalPromptSchema
    relation_pool: RelationPool
    name_pool_name: str


# -------------------------- Family Trees -------------------------- 

from .prompt_definitions.family_trees import (
    ABC_family_tree_who_is_schema,
    ABC_family_tree_yes_no_schema,
    family_relations,
)

ABC_FAMILY_TREE_WHO_IS_CONFIG = PromptConfig(
    name = 'ABC_family_tree_who_is',
    prompt_schema = ABC_family_tree_who_is_schema,
    relation_pool = family_relations,
    name_pool_name = 'human_names'
)

ABC_FAMILY_TREE_YES_NO_CONFIG = PromptConfig(
    name = 'ABC_family_tree_yes_no',
    prompt_schema = ABC_family_tree_yes_no_schema,
    relation_pool = family_relations,
    name_pool_name = 'human_names'
)

ABC_ALIEN_FAMILY_TREE_WHO_IS_CONFIG = PromptConfig(
    name = 'ABC_alien_family_tree_who_is',
    prompt_schema = ABC_family_tree_who_is_schema,
    relation_pool = family_relations,
    name_pool_name = 'alien_names'
)

ABC_ALIEN_FAMILY_TREE_YES_NO_CONFIG = PromptConfig(
    name = 'ABC_alien_family_tree_yes_no',
    prompt_schema = ABC_family_tree_yes_no_schema,
    relation_pool = family_relations,
    name_pool_name = 'alien_names'
)


# ----------------------------- Biology ---------------------------- 

from .prompt_definitions.biology import (
    ABC_gene_regulatory_network_schema,
    ABC_neural_circuit_schema,
    activation_repression_relations,
    excitation_inhibition_relations,
)

ABC_GENE_REGULATORY_NETWORK_CONFIG = PromptConfig(
    name = 'ABC_gene_regulatory_network',
    prompt_schema = ABC_gene_regulatory_network_schema,
    relation_pool = activation_repression_relations,
    name_pool_name = 'alien_names'
)

ABC_NEURAL_CIRCUIT_CONFIG = PromptConfig(
    name = 'ABC_neural_circuit',
    prompt_schema = ABC_neural_circuit_schema,
    relation_pool = excitation_inhibition_relations,
    name_pool_name = 'alien_names'
)


# ---------------------- Transitive Inference ---------------------- 

from .prompt_definitions.transitive_inference import (
    ABC_transitive_inference_schema,
    older_younger_relations,
    taller_shorter_relations,
    faster_slower_relations,
)

ABC_OLDER_YOUNGER_CONFIG = PromptConfig(
    name = 'ABC_older_younger',
    prompt_schema = ABC_transitive_inference_schema,
    relation_pool = older_younger_relations,
    name_pool_name = 'human_names'
)

ABC_TALLER_SHORTER_CONFIG = PromptConfig(
    name = 'ABC_taller_shorter',
    prompt_schema = ABC_transitive_inference_schema,
    relation_pool = taller_shorter_relations,
    name_pool_name = 'human_names'
)

ABC_FASTER_SLOWER_CONFIG = PromptConfig(
    name = 'ABC_faster_slower',
    prompt_schema = ABC_transitive_inference_schema,
    relation_pool = faster_slower_relations,
    name_pool_name = 'human_names'
)

ABC_ALIEN_OLDER_YOUNGER_CONFIG = PromptConfig(
    name = 'ABC_alien_older_younger',
    prompt_schema = ABC_transitive_inference_schema,
    relation_pool = older_younger_relations,
    name_pool_name = 'alien_names'
)

ABC_ALIEN_TALLER_SHORTER_CONFIG = PromptConfig(
    name = 'ABC_alien_taller_shorter',
    prompt_schema = ABC_transitive_inference_schema,
    relation_pool = taller_shorter_relations,
    name_pool_name = 'alien_names'
)

ABC_ALIEN_FASTER_SLOWER_CONFIG = PromptConfig(
    name = 'ABC_alien_faster_slower',
    prompt_schema = ABC_transitive_inference_schema,
    relation_pool = faster_slower_relations,
    name_pool_name = 'alien_names'
)


# ---------------------------- Companies --------------------------- 

from .prompt_definitions.companies import (
    ABC_companies_schema,
    companies_relations,
)

ABC_COMPANY_STRUCTURE_CONFIG = PromptConfig(
    name = 'ABC_company_structure',
    prompt_schema = ABC_companies_schema,
    relation_pool = companies_relations,
    name_pool_name = 'human_names'
)

ABC_ALIEN_COMPANY_STRUCTURE_CONFIG = PromptConfig(
    name = 'ABC_alien_company_structure',
    prompt_schema = ABC_companies_schema,
    relation_pool = companies_relations,
    name_pool_name = 'alien_names'
)


# -------------------------- Equivalences -------------------------- 

from .prompt_definitions.equivalences import (
    ABC_alien_proposition_equivalence_schema,
    ABC_alien_variable_equivalence_schema,
    ABC_proposition_equivalence_schema,
    ABC_variable_equivalence_schema,
    proposition_equivalence_relations,
    variable_equivalence_relations,
)

ABC_PROPOSITION_EQUIVALENCE_CONFIG = PromptConfig(
    name = 'ABC_proposition_equivalence',
    prompt_schema = ABC_proposition_equivalence_schema,
    relation_pool = proposition_equivalence_relations,
    name_pool_name = 'numbers'
)

ABC_ALIEN_PROPOSITION_EQUIVALENCE_CONFIG = PromptConfig(
    name = 'ABC_alien_proposition_equivalence',
    prompt_schema = ABC_alien_proposition_equivalence_schema,
    relation_pool = proposition_equivalence_relations,
    name_pool_name = 'alien_names'
)

ABC_VARIABLE_EQUIVALENCE_CONFIG = PromptConfig(
    name = 'ABC_variable_equivalence',
    prompt_schema = ABC_variable_equivalence_schema,
    relation_pool = variable_equivalence_relations,
    name_pool_name = 'letters'
)

ABC_ALIEN_VARIABLE_EQUIVALENCE_CONFIG = PromptConfig(
    name = 'ABC_alien_variable_equivalence',
    prompt_schema = ABC_alien_variable_equivalence_schema,
    relation_pool = variable_equivalence_relations,
    name_pool_name = 'alien_names'
)


# --------------------- Spatial Relationships ---------------------- 

from .prompt_definitions.spatial_relationships import (
    ABC_spatial_relationships_schema,
    east_west_relations,
    north_south_relations,
    ahead_behind_relations,
)

ABC_EAST_WEST_CONFIG = PromptConfig(
    name = 'ABC_east_west',
    prompt_schema = ABC_spatial_relationships_schema,
    relation_pool = east_west_relations,
    name_pool_name = 'places'
)

ABC_ALIEN_EAST_WEST_CONFIG = PromptConfig(
    name = 'ABC_alien_east_west',
    prompt_schema = ABC_spatial_relationships_schema,
    relation_pool = east_west_relations,
    name_pool_name = 'alien_names'
)

ABC_NORTH_SOUTH_CONFIG = PromptConfig(
    name = 'ABC_north_south',
    prompt_schema = ABC_spatial_relationships_schema,
    relation_pool = north_south_relations,
    name_pool_name = 'places'
)

ABC_ALIEN_NORTH_SOUTH_CONFIG = PromptConfig(
    name = 'ABC_alien_north_south',
    prompt_schema = ABC_spatial_relationships_schema,
    relation_pool = north_south_relations,
    name_pool_name = 'alien_names'
)

ABC_AHEAD_BEHIND_CONFIG = PromptConfig(
    name = 'ABC_ahead_behind',
    prompt_schema = ABC_spatial_relationships_schema,
    relation_pool = ahead_behind_relations,
    name_pool_name = 'places'
)

ABC_ALIEN_AHEAD_BEHIND_CONFIG = PromptConfig(
    name = 'ABC_alien_ahead_behind',
    prompt_schema = ABC_spatial_relationships_schema,
    relation_pool = ahead_behind_relations,
    name_pool_name = 'alien_names'
)


# ----------------- Additional "map vs. graph" prompts ---------------

from .prompt_definitions.additional_map_vs_graph_prompts import (
    AB_CD_family_tree_who_is_schema,
    AB_CD_family_tree_yes_no_schema,
    ABC_family_tree_descendant_ancestor_schema,
    ABC_older_younger_unique_location_who_is_schema,
    ABC_older_younger_unique_location_yes_no_schema,
    descendant_ancestor_relations,
    family_relations_simple_queries,
    unique_older_younger_relations,
)

AB_CD_FAMILY_TREES_WHO_IS_CONFIG = PromptConfig(
    name = 'AB_CD_family_trees_who_is',
    prompt_schema = AB_CD_family_tree_who_is_schema,
    relation_pool = family_relations_simple_queries,
    name_pool_name = 'human_names'
)

AB_CD_FAMILY_TREES_YES_NO_CONFIG = PromptConfig(
    name = 'AB_CD_family_trees_yes_no',
    prompt_schema = AB_CD_family_tree_yes_no_schema,
    relation_pool = family_relations_simple_queries,
    name_pool_name = 'human_names'
)

ABC_FAMILY_TREES_DESCENDANT_ANCESTOR_CONFIG = PromptConfig(
    name = 'ABC_family_trees_descendant_ancestor',
    prompt_schema = ABC_family_tree_descendant_ancestor_schema,
    relation_pool = descendant_ancestor_relations,
    name_pool_name = 'human_names'
)

ABC_ALIEN_FAMILY_TREES_DESCENDANT_ANCESTOR_CONFIG = PromptConfig(
    name = 'ABC_alien_family_trees_descendant_ancestor',
    prompt_schema = ABC_family_tree_descendant_ancestor_schema,
    relation_pool = descendant_ancestor_relations,
    name_pool_name = 'alien_names'
)

ABC_OLDER_YOUNGER_UNIQUE_LOCATIONS_WHO_IS_CONFIG = PromptConfig(
    name = 'ABC_older_younger_unique_locations_who_is',
    prompt_schema = ABC_older_younger_unique_location_who_is_schema,
    relation_pool = unique_older_younger_relations,
    name_pool_name = 'human_names'
)

ABC_OLDER_YOUNGER_UNIQUE_LOCATIONS_YES_NO_CONFIG = PromptConfig(
    name = 'ABC_older_younger_unique_locations_yes_no',
    prompt_schema = ABC_older_younger_unique_location_yes_no_schema,
    relation_pool = unique_older_younger_relations,
    name_pool_name = 'human_names'
)

ABC_ALIEN_OLDER_YOUNGER_UNIQUE_LOCATIONS_YES_NO_CONFIG = PromptConfig(
    name = 'ABC_alien_older_younger_unique_locations_yes_no',
    prompt_schema = ABC_older_younger_unique_location_yes_no_schema,
    relation_pool = unique_older_younger_relations,
    name_pool_name = 'alien_names'
)



# ------------------------------------------------------------------- 

PROMPT_CONFIGS = {
    config.name: config 
    for config in 
    [
        ABC_FAMILY_TREE_WHO_IS_CONFIG,
        ABC_FAMILY_TREE_YES_NO_CONFIG,
        ABC_ALIEN_FAMILY_TREE_WHO_IS_CONFIG,
        ABC_ALIEN_FAMILY_TREE_YES_NO_CONFIG,
        ABC_OLDER_YOUNGER_CONFIG,
        ABC_TALLER_SHORTER_CONFIG,
        ABC_FASTER_SLOWER_CONFIG,
        ABC_ALIEN_OLDER_YOUNGER_CONFIG,
        ABC_ALIEN_TALLER_SHORTER_CONFIG,
        ABC_ALIEN_FASTER_SLOWER_CONFIG,
        ABC_COMPANY_STRUCTURE_CONFIG,
        ABC_ALIEN_COMPANY_STRUCTURE_CONFIG,
        ABC_GENE_REGULATORY_NETWORK_CONFIG,
        ABC_NEURAL_CIRCUIT_CONFIG,
        ABC_PROPOSITION_EQUIVALENCE_CONFIG,
        ABC_ALIEN_PROPOSITION_EQUIVALENCE_CONFIG,
        ABC_VARIABLE_EQUIVALENCE_CONFIG,
        ABC_ALIEN_VARIABLE_EQUIVALENCE_CONFIG,
        ABC_EAST_WEST_CONFIG,
        ABC_ALIEN_EAST_WEST_CONFIG,
        ABC_NORTH_SOUTH_CONFIG,
        ABC_ALIEN_NORTH_SOUTH_CONFIG,
        ABC_AHEAD_BEHIND_CONFIG,
        ABC_ALIEN_AHEAD_BEHIND_CONFIG,
        AB_CD_FAMILY_TREES_WHO_IS_CONFIG,
        AB_CD_FAMILY_TREES_YES_NO_CONFIG,
        ABC_FAMILY_TREES_DESCENDANT_ANCESTOR_CONFIG,
        ABC_ALIEN_FAMILY_TREES_DESCENDANT_ANCESTOR_CONFIG,
        ABC_OLDER_YOUNGER_UNIQUE_LOCATIONS_WHO_IS_CONFIG,
        ABC_OLDER_YOUNGER_UNIQUE_LOCATIONS_YES_NO_CONFIG,
        ABC_ALIEN_OLDER_YOUNGER_UNIQUE_LOCATIONS_YES_NO_CONFIG,
    ]
}
