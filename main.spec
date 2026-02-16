# -*- mode: python ; coding: utf-8 -*-

import sys
import os

# Добавляем все пути к модулям
sys.path.insert(0, os.path.abspath('.'))

a = Analysis(
    ['main.py'],
    pathex=['.', './analysis', './api', './shared'],
    binaries=[],
    datas=[
        ('icon.png', '.'),
        # Добавьте сюда другие нужные файлы
    ],
    hiddenimports=[
        'analysis.analysis_stocks.tree_solver_analysis.tree_solver_analysis',
        'analysis.analysis_stocks.ai_analysis.ai_anomaly_detector_ae.ai_anomaly_detector_ae',
        'analysis.analysis_stocks.ai_analysis.ai_risk_analysis.ai_risk_analysis',
        'analysis.analysis_stocks.cluster_analysis.cluster_analysis',
        'analysis.analysis_stocks.regression_analysis.regression_analysis',
        'api.t_invest_api',
        'shared.config',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    noarchive=False,
)

# Добавляем все подпакеты
for root, dirs, files in os.walk('analysis'):
    for file in files:
        if file.endswith('.py'):
            module_path = os.path.join(root, file)
            module_name = module_path.replace(os.sep, '.')[:-3]
            if module_name not in a.hiddenimports:
                a.hiddenimports.append(module_name)

for root, dirs, files in os.walk('api'):
    for file in files:
        if file.endswith('.py'):
            module_path = os.path.join(root, file)
            module_name = module_path.replace(os.sep, '.')[:-3]
            if module_name not in a.hiddenimports:
                a.hiddenimports.append(module_name)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AnalyticSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # False - без консоли, True - с консолью (для отладки)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.png'
)