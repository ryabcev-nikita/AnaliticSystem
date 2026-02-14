# ==================== КОНФИГУРАЦИЯ ПУТЕЙ ====================


import os
from ai_anomaly_detector_models.ai_anomaly_constants.ai_anomaly_constants import (
    AE_FILES,
)


class AEPathConfig:
    """Конфигурация путей к файлам для автоэнкодера"""

    @staticmethod
    def setup_directories():
        def find_root_dir(marker=".gitignore"):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            while True:
                if os.path.exists(os.path.join(current_dir, marker)):
                    return current_dir
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # Достигли корня файловой системы
                    return None
                current_dir = parent_dir

        """Создание необходимых директорий"""
        parent_dir = find_root_dir()
        nn_ae_detector_dir = f"{parent_dir}/data/nn_ae_anomaly_detector"
        os.makedirs(nn_ae_detector_dir, exist_ok=True)

        return {
            "parent_dir": parent_dir,
            "nn_ae_detector_dir": nn_ae_detector_dir,
            "input_file": f"{parent_dir}/data/fundamentals_shares.xlsx",
            "output_file": f"{nn_ae_detector_dir}/{AE_FILES.AE_PORTFOLIO_RESULTS}",
            "ae_anomaly_file": f"{nn_ae_detector_dir}/{AE_FILES.AE_ANOMALY_ANALYSIS}",
            "ae_portfolio_comparison": f"{nn_ae_detector_dir}/{AE_FILES.AE_PORTFOLIO_COMPARISON}",
            "ae_portfolio_optimal": f"{nn_ae_detector_dir}/{AE_FILES.AE_PORTFOLIO_OPTIMAL}",
            "ae_portfolio_summary": f"{nn_ae_detector_dir}/{AE_FILES.AE_PORTFOLIO_SUMMARY}",
        }


AE_PATHS = AEPathConfig.setup_directories()
