"""Simulation layer: Habitat wrapper, embodiment, drive/physics, sensors, noise models.

habitat_sim is imported lazily inside the modules that need it, so the rest of the
package (metrics, language, policy) stays importable and testable without Habitat.
"""
