# ==================== КОНФИГУРАЦИЯ ПУТЕЙ ====================


import os
from ai_risk_models.ai_risk_constants.ai_risk_constants import NN_FILES


class NNRiskPathConfig:
    """Конфигурация путей к файлам для нейросетевого анализа рисков"""

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
        nn_risk_analyzer_dir = f"{parent_dir}/data/nn_risk_analyzer"
        os.makedirs(nn_risk_analyzer_dir, exist_ok=True)

        return {
            "parent_dir": parent_dir,
            "nn_risk_analyzer_dir": nn_risk_analyzer_dir,
            "input_file": f"{parent_dir}/data/fundamentals_shares.xlsx",
            "nn_risk_portfolio_base": f"{nn_risk_analyzer_dir}/{NN_FILES.NN_RISK_PORTFOLIO_BASE}",
            "nn_risk_efficient_frontier": f"{nn_risk_analyzer_dir}/{NN_FILES.NN_RISK_EFFICIENT_FRONTIER}",
            "nn_risk_portfolio_comparison": f"{nn_risk_analyzer_dir}/{NN_FILES.NN_RISK_PORTFOLIO_COMPARISON}",
            "nn_risk_portfolio_results": f"{nn_risk_analyzer_dir}/{NN_FILES.NN_RISK_PORTFOLIO_RESULTS}",
        }


NN_RISK_PATHS = NNRiskPathConfig.setup_directories()
