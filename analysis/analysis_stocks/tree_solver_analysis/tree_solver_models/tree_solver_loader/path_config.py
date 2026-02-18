# ==================== КОНФИГУРАЦИЯ ПУТЕЙ ====================
import os
from ...tree_solver_models.tree_solver_constants.tree_solver_constants import (
    FILE_CONSTANTS,
)


class PathConfig:
    """Конфигурация путей к файлам"""

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
        tree_solver_dir = f"{parent_dir}/data/tree_solver_analysis"
        os.makedirs(tree_solver_dir, exist_ok=True)

        return {
            "parent_dir": parent_dir,
            "tree_solver_dir": tree_solver_dir,
            "file_path": f"{parent_dir}/data/fundamentals_shares.xlsx",
            "decision_tree": f"{tree_solver_dir}/{FILE_CONSTANTS.DECISION_TREE_FILE}",
            "efficient_frontier": f"{tree_solver_dir}/{FILE_CONSTANTS.EFFICIENT_FRONTIER_FILE}",
            "portfolio_report": f"{tree_solver_dir}/{FILE_CONSTANTS.INVEST_PORTFOLIO_REPORT}",
            "optimal_portfolio": f"{tree_solver_dir}/{FILE_CONSTANTS.OPTIMAL_PORTFOLIO_FILE}",
        }


PATHS = PathConfig.setup_directories()
