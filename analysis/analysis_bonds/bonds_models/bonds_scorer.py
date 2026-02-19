from typing import List

from bonds_models.bonds_constants import (
    CURRENCY_PARAMS,
    OPTIMIZATION_PARAMS,
    SCORING_WEIGHTS,
    SECTORS,
)
from bonds_models.bonds_models import Bond


class BondScorer:
    """Класс для скоринга облигаций"""

    def __init__(self, bonds_list: List[Bond]):
        self.bonds_list = bonds_list
        self.calculate_scores()

    def calculate_scores(self):
        """Расчет скоринговых оценок"""

        # Нормализация доходности
        yields = [b.current_yield for b in self.bonds_list]
        min_yield, max_yield = min(yields), max(yields)
        yield_range = max_yield - min_yield if max_yield > min_yield else 1

        # Нормализация дюрации
        durations = [b.modified_duration for b in self.bonds_list]
        min_dur, max_dur = min(durations), max(durations)
        dur_range = max_dur - min_dur if max_dur > min_dur else 1

        for bond in self.bonds_list:
            # 1. Score по доходности (чем выше, тем лучше)
            bond.yield_score = (bond.current_yield - min_yield) / yield_range

            # 2. Score по риску (чем ниже риск, тем лучше)
            bond.risk_score = 1 - (bond.risk_level / 3)

            # 3. Score по ликвидности
            bond.liquidity_score_norm = bond.liquidity_score

            # 4. Score по дюрации (для target duration)
            target_dur = OPTIMIZATION_PARAMS["target_duration"]
            bond.duration_score = 1 - min(
                abs(bond.modified_duration - target_dur) / target_dur, 1
            )

            # 5. Score по сектору
            sector_weight = SECTORS.get(bond.sector, {}).get("max_weight", 0.1)
            bond.sector_score = min(sector_weight * 5, 1)  # Нормализация

            # 6. Score по валюте
            currency_params = CURRENCY_PARAMS.get(bond.currency, CURRENCY_PARAMS["rub"])
            if bond.current_yield >= currency_params["min_yield"]:
                bond.currency_score = 1
            else:
                bond.currency_score = bond.current_yield / currency_params["min_yield"]

            # Общий score
            bond.total_score = (
                SCORING_WEIGHTS["yield_score"] * bond.yield_score
                + SCORING_WEIGHTS["risk_score"] * bond.risk_score
                + SCORING_WEIGHTS["liquidity_score"] * bond.liquidity_score_norm
                + SCORING_WEIGHTS["duration_score"] * bond.duration_score
                + SCORING_WEIGHTS["sector_score"] * bond.sector_score
                + SCORING_WEIGHTS["currency_score"] * bond.currency_score
            )

    def get_top_bonds(self, n: int = 50) -> List[Bond]:
        """Получение топ-N облигаций по скорингу"""
        sorted_bonds = sorted(
            self.bonds_list, key=lambda x: x.total_score, reverse=True
        )
        return sorted_bonds[:n]

    def get_bonds_by_risk_level(self, risk_level: int) -> List[Bond]:
        """Получение облигаций с определенным уровнем риска"""
        return [b for b in self.bonds_list if b.risk_level == risk_level]
