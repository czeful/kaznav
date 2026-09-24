"""kznav — zero-shot object-goal navigation bench with swappable VLM backends.

Layers (imports only flow downwards; navigation logic never imports a concrete model):

    run/        episode + matrix runners, logging, manifests
    policy/     hypothesis belief, stop criterion, frontier choice, planning, control
    perception/ occupancy, frontiers, detector / segmenter / value-encoder interfaces
    language/   VLM backend interface, registry, prompts, schemas
    sim/        simulator + embodiment (Habitat is imported lazily)
    eval/       metrics and statistics (pure functions, no simulator dependency)
    report/     tables, plots, videos
"""

__version__ = "0.1.0"
