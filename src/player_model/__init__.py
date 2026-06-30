"""HEATER Layer-0 shared player model (Advanced Value Engine, Phase 1).

Produces, per player per scoring category, a posterior {mean, sigma2, tau2} with a
distributional margin descriptor, plus an availability survival distribution and a
display-only G-score. The single source of truth feeding every surface.
"""

from src.player_model.availability import AvailabilitySurvival, availability_survival, sample_active_weeks
from src.player_model.gscore import LeagueContext, category_gscore, player_gscore
from src.player_model.model import (
    PlayerModel,
    build_league_context,
    build_player_model,
    build_player_models,
)
from src.player_model.posterior import CategoryPosterior, category_posterior, player_posteriors

__all__ = [
    "AvailabilitySurvival",
    "CategoryPosterior",
    "LeagueContext",
    "PlayerModel",
    "availability_survival",
    "build_league_context",
    "build_player_model",
    "build_player_models",
    "category_gscore",
    "category_posterior",
    "player_gscore",
    "player_posteriors",
    "sample_active_weeks",
]
