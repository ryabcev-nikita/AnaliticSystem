# ==================== КОНФИГУРАЦИЯ ПУТЕЙ ====================


import os

from cluster_models.cluster_constants.cluster_constants import CLUSTER_FILES


class ClusterPathConfig:
    """Конфигурация путей к файлам для кластерного анализа"""

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
        cluster_dir = f"{parent_dir}/data/cluster_analysis"
        os.makedirs(cluster_dir, exist_ok=True)

        return {
            "cluster_dir": cluster_dir,
            "data_path": f"{parent_dir}/data/fundamentals_shares.xlsx",
            "cluster_analysis": f"{cluster_dir}/{CLUSTER_FILES.CLUSTER_ANALYSIS_FILE}",
            "cluster_optimization": f"{cluster_dir}/{CLUSTER_FILES.CLUSTER_OPTIMIZATION_FILE}",
            "portfolio_comparison": f"{cluster_dir}/{CLUSTER_FILES.PORTFOLIO_COMPARISON_FILE}",
            "cluster_allocation": f"{cluster_dir}/{CLUSTER_FILES.CLUSTER_ALLOCATION_FILE}",
            "clustered_companies": f"{cluster_dir}/{CLUSTER_FILES.CLUSTERED_COMPANIES_FILE}",
            "investment_cluster_report": f"{cluster_dir}/{CLUSTER_FILES.INVESTMENT_CLUSTER_REPORT}",
        }


CLUSTER_PATHS = ClusterPathConfig.setup_directories()
