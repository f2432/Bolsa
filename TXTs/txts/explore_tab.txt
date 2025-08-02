import os
import pandas as pd
import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
    QMessageBox, QComboBox, QLabel, QHBoxLayout, QProgressBar, QFileDialog, QTabWidget, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from gui.suggestion_detail_dialog import SuggestionDetailDialog
from universe_utils import UNIVERSE_FUNCS
from ai.predictor import AIPredictor

SUGGESTION_CACHE_DIR = "suggestion_cache"
os.makedirs(SUGGESTION_CACHE_DIR, exist_ok=True)

class SuggestionWorker(QThread):
    progress = pyqtSignal(int)
    row_ready = pyqtSignal(int, dict)
    finished = pyqtSignal(list)

    def __init__(self, tickers, data_provider, predictor, portfolio_manager):
        super().__init__()
        self.tickers = tickers
        self.data_provider = data_provider
        self.predictor = predictor
        self.portfolio_manager = portfolio_manager

    def run(self):
        total = len(self.tickers)
        linhas_cache = []
        for i, ticker in enumerate(self.tickers):
            res = {"Ticker": ticker}
            try:
                data = self.data_provider.get_historical_data(ticker)
                # Preço atual:
                try:
                    preco_atual = float(self.data_provider.get_current_price(ticker))
                except Exception:
                    preco_atual = float('nan')
                res["PrecoAtual"] = preco_atual
                if data is None or data.empty or 'Close' not in data.columns:
                    for m in ["Logistic", "RF", "MLP"]:
                        for h in [1, 3]:
                            res[f"ProbSubida_{m}_{h}d"] = float('nan')
                            res[f"PrecoPrev_{m}_{h}d"] = float('nan')
                            res[f"Sugestao_{m}_{h}d"] = ""
                            res[f"Features_{m}_{h}d"] = ""
                else:
                    for model, m_key in [("logistic", "Logistic"), ("rf", "RF"), ("mlp", "MLP")]:
                        for n_ahead in [1, 3]:
                            try:
                                predictor = AIPredictor(model_type=model, n_ahead=n_ahead)
                                predictor.train_on_data(data, model_type=model)
                                proba = predictor.predict_proba(data)
                                price_pred = predictor.predict_price(data)
                                features = predictor.get_last_features(data)
                                sug = self.gerar_sugestao(proba)
                            except Exception as e:
                                proba = [float('nan'), float('nan')]
                                price_pred = float('nan')
                                sug = ""
                                features = {}
                            res[f"ProbSubida_{m_key}_{n_ahead}d"] = float(proba[1]) if proba is not None and len(proba) > 1 else float('nan')
                            res[f"PrecoPrev_{m_key}_{n_ahead}d"] = float(price_pred) if price_pred is not None else float('nan')
                            res[f"Sugestao_{m_key}_{n_ahead}d"] = sug
                            res[f"Features_{m_key}_{n_ahead}d"] = str(features) if features else ""
                for n_ahead in [1, 3]:
                    probs = [res.get(f"ProbSubida_{m}_{n_ahead}d", float('nan')) for m in ["Logistic", "RF", "MLP"]]
                    precos = [res.get(f"PrecoPrev_{m}_{n_ahead}d", float('nan')) for m in ["Logistic", "RF", "MLP"]]
                    res[f"Consenso_ProbSubida_{n_ahead}d"] = float(pd.Series(probs).mean(skipna=True))
                    res[f"Consenso_PrecoPrev_{n_ahead}d"] = float(pd.Series(precos).mean(skipna=True))
            except Exception as err:
                print(f"[ERRO SUGESTÃO] {ticker}: {err}")
            self.row_ready.emit(i, res)
            linhas_cache.append(res)
            self.progress.emit(int((i + 1) / total * 100))
        self.finished.emit(linhas_cache)

    def gerar_sugestao(self, proba):
        if proba is None or len(proba) < 2:
            return ""
        if proba[1] > 0.65:
            return "Potencial Compra"
        elif proba[1] < 0.35:
            return "Evitar"
        else:
            return "Neutro"

class ExploreTab(QWidget):
    def __init__(self, data_provider, predictor, portfolio_manager, indicator_analysis_text, parent=None):
        super().__init__(parent)
        self.data_provider = data_provider
        self.predictor = predictor
        self.portfolio_manager = portfolio_manager
        self.indicator_analysis_text = indicator_analysis_text
        self.parent = parent
        self.universe_last_scan = {}
        # Cache para fiabilidade (Sharpe ratio normalizado) por ticker
        self.reliability_cache = {}

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        

        universe_panel = QHBoxLayout()
        self.universo_combo = QComboBox()
        self.universo_combo.addItems(list(UNIVERSE_FUNCS.keys()))
        universe_panel.addWidget(QLabel("Bolsa/Índice:"))
        universe_panel.addWidget(self.universo_combo)
        self.export_btn = QPushButton("Exportar cache")
        self.export_btn.clicked.connect(self.exportar_cache)
        self.import_btn = QPushButton("Importar cache")
        self.import_btn.clicked.connect(self.importar_cache)
        universe_panel.addWidget(self.export_btn)
        universe_panel.addWidget(self.import_btn)
        self.last_scan_label = QLabel()
        self.layout.addWidget(self.last_scan_label)
        self.layout.addLayout(universe_panel)
        self.universo_combo.currentTextChanged.connect(self.on_universo_trocado)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        self.suggestions_table = QTableWidget()
        self.setup_suggestions_table()
        self.tabs.addTab(self.suggestions_table, "Todos")
        self.top25_table = QTableWidget()
        self.setup_top25_table()
        self.tabs.addTab(self.top25_table, "Top 25 Consenso")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)
        refresh_sug_btn = QPushButton("Atualizar Sugestões")
        refresh_sug_btn.clicked.connect(self.atualizar_sugestoes)
        self.layout.addWidget(refresh_sug_btn)
        self.mostrar_cache()

    def on_universo_trocado(self, *args):
        self.mostrar_cache()
        self.update_last_scan_label()

    def setup_suggestions_table(self):
        headers = [
            "Ticker", "Preço Atual",
            "ProbSubida Logistic 1d", "Sugestão Logistic 1d",
            "ProbSubida Logistic 3d", "Sugestão Logistic 3d",
            "ProbSubida RF 1d", "Sugestão RF 1d",
            "ProbSubida RF 3d", "Sugestão RF 3d",
            "ProbSubida MLP 1d", "Sugestão MLP 1d",
            "ProbSubida MLP 3d", "Sugestão MLP 3d",
            "Consenso ProbSubida 1d", "Consenso ProbSubida 3d",
            "Ver Detalhe"
        ]
        self.suggestions_table.setColumnCount(len(headers))
        self.suggestions_table.setHorizontalHeaderLabels(headers)
        self.suggestions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def setup_top25_table(self):
        headers = [
            "Ticker", "Preço Atual", "Consenso ProbSubida 1d", "Consenso PrecoPrev 1d",
            "Consenso ProbSubida 3d", "Consenso PrecoPrev 3d", "Ver Detalhe"
        ]
        self.top25_table.setColumnCount(len(headers))
        self.top25_table.setHorizontalHeaderLabels(headers)
        self.top25_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def update_last_scan_label(self):
        universo_nome = self.universo_combo.currentText()
        dt = self.universe_last_scan.get(universo_nome, None)
        if dt:
            texto = f"Última análise deste universo: {dt.strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            texto = "Ainda não foi feita análise neste universo."
        self.last_scan_label.setText(texto)

    def get_universe(self, nome):
        if nome in UNIVERSE_FUNCS:
            try:
                return UNIVERSE_FUNCS[nome]()
            except Exception as e:
                QMessageBox.warning(self, "Erro a carregar universo", f"Erro ao carregar {nome}: {e}")
                return []
        return []

    def cache_file(self, universe_name):
        return os.path.join(SUGGESTION_CACHE_DIR, f"suggestions_{universe_name}.csv")

    def mostrar_cache(self):
        universo_nome = self.universo_combo.currentText()
        cache_f = self.cache_file(universo_nome)
        if os.path.exists(cache_f):
            df = pd.read_csv(cache_f, na_values=["nan", "NaN", ""])
            for col in df.columns:
                if any(x in col for x in [
                    "ProbSubida", "PrecoPrev", "Consenso_ProbSubida", "Consenso_PrecoPrev", "PrecoAtual"
                ]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            self.preencher_tabela(df)
        else:
            self.suggestions_table.setRowCount(0)
            self.top25_table.setRowCount(0)

    def preencher_tabela(self, df):
        # Calcula uma pontuação combinada para ordenar tickers
        # A pontuação baseia-se na probabilidade de subida (2 dias)
        # multiplicada por um factor de valorização previsto (preço futuro vs atual)
        try:
            df = df.copy()
            # Evita divisão por zero e valores nulos
            df["Score"] = float('nan')
            for idx, row in df.iterrows():
                # Usa 2d se existir, caso contrário recorre a 1d
                prob2 = row.get("Consenso_ProbSubida_2d", row.get("Consenso_ProbSubida_1d", float('nan')))
                preco_atual = row.get("PrecoAtual", float('nan'))
                preco_prev = row.get("Consenso_PrecoPrev_2d", row.get("Consenso_PrecoPrev_1d", float('nan')))
                # Calcula fiabilidade (Sharpe normalizado) se ainda não estiver no cache
                ticker = row.get("Ticker")
                reliability = None
                if ticker:
                    reliability = self.reliability_cache.get(ticker)
                    if reliability is None:
                        try:
                            # Faz backtest simples para medir Sharpe do SMA Crossover
                            from strategies.sma_crossover import SMACrossoverStrategy
                            from backtest.backtester import Backtester
                            from backtest.metrics import calculate_metrics
                            data_hist = self.data_provider.get_historical_data(ticker)
                            if data_hist is not None and not data_hist.empty and 'Close' in data_hist.columns:
                                strategy = SMACrossoverStrategy(short_window=50, long_window=200)
                                signals_bt = strategy.generate_signals(data_hist)
                                bt = Backtester(initial_capital=10000)
                                results_bt = bt.run(data_hist, signals_bt)
                                eq_bt = results_bt['equity_curve']
                                trades_bt = results_bt.get('trades')
                                metrics_bt = calculate_metrics(data_hist, signals_bt, eq_bt, trades_bt)
                                sharpe_val = float(metrics_bt.get('sharpe', 0) or 0)
                                # Normaliza Sharpe para [0,1] usando função suave (e.g. tanh)
                                import math
                                reliability = math.tanh(max(sharpe_val, 0))
                            else:
                                reliability = 0
                        except Exception:
                            reliability = 0
                        self.reliability_cache[ticker] = reliability
                score = float('nan')
                try:
                    if pd.notna(prob2):
                        price_diff = 0.0
                        if pd.notna(preco_atual) and preco_atual > 0 and pd.notna(preco_prev):
                            price_diff = (preco_prev - preco_atual) / preco_atual
                        # Apenas considera price_diff positivo
                        if price_diff < 0:
                            price_diff = 0
                        # Incorpora fiabilidade (aditivo) se calculada
                        rel_factor = reliability if reliability is not None else 0
                        score = prob2 * (1 + price_diff) * (1 + rel_factor)
                except Exception:
                    pass
                df.at[idx, "Score"] = score
        except Exception:
            pass
        self.suggestions_table.setRowCount(len(df))
        for i, row in df.iterrows():
            self.suggestions_table.setItem(i, 0, QTableWidgetItem(str(row["Ticker"])))
            self.suggestions_table.setItem(i, 1, QTableWidgetItem(f'{row.get("PrecoAtual", float("nan")):.2f}' if not pd.isna(row.get("PrecoAtual", float('nan'))) else "N/A"))
            col = 2
            for m in ["Logistic", "RF", "MLP"]:
                for h in [1, 3]:
                    val = row.get(f"ProbSubida_{m}_{h}d", float('nan'))
                    try:
                        valf = float(val)
                        txt = f"{valf:.2%}" if not pd.isna(valf) else ""
                    except Exception:
                        txt = ""
                    self.suggestions_table.setItem(i, col, QTableWidgetItem(txt)); col += 1
                    sug = row.get(f"Sugestao_{m}_{h}d", "")
                    self.suggestions_table.setItem(i, col, QTableWidgetItem(str(sug))); col += 1
            for h in [1, 3]:
                val = row.get(f"Consenso_ProbSubida_{h}d", float('nan'))
                try:
                    valf = float(val)
                    txt = f"{valf:.2%}" if not pd.isna(valf) else ""
                except Exception:
                    txt = ""
                self.suggestions_table.setItem(i, col, QTableWidgetItem(txt)); col += 1
            # Botão "Ver Detalhe" SEMPRE na última coluna!
            btn = QPushButton("Ver Detalhe")
            btn.clicked.connect(lambda _, t=row["Ticker"]: self.mostrar_detalhe_acao(t))
            self.suggestions_table.setCellWidget(i, self.suggestions_table.columnCount() - 1, btn)

        # Top 25
        if "Consenso_ProbSubida_1d" not in df.columns:
            self.top25_table.setRowCount(0)
            return

        df_top = df.copy()
        # Usa a pontuação calculada (Score) se existir; caso contrário, recorre à probabilidade
        if "Score" in df_top.columns:
            df_top = df_top[pd.notna(df_top["Score"])]
            df_top = df_top.sort_values("Score", ascending=False).head(25)
        else:
            # Usa probabilidade 2d ou 1d conforme disponível
            col_prob = "Consenso_ProbSubida_2d" if "Consenso_ProbSubida_2d" in df_top.columns else "Consenso_ProbSubida_1d"
            df_top[col_prob] = pd.to_numeric(df_top[col_prob], errors="coerce")
            df_top = df_top[df_top[col_prob].notna() & (df_top[col_prob] > 0.001)]
            df_top = df_top.sort_values(col_prob, ascending=False).head(25)
        self.top25_table.setRowCount(25)
        for i in range(25):
            if i < len(df_top):
                row = df_top.iloc[i]
                self.top25_table.setItem(i, 0, QTableWidgetItem(str(row["Ticker"])))
                self.top25_table.setItem(i, 1, QTableWidgetItem(f'{row.get("PrecoAtual", float("nan")):.2f}' if not pd.isna(row.get("PrecoAtual", float('nan'))) else "N/A"))
                for j, colname in enumerate([
                    "Consenso_ProbSubida_1d", "Consenso_PrecoPrev_1d", "Consenso_ProbSubida_3d", "Consenso_PrecoPrev_3d"
                ], start=2):
                    val = row.get(colname, float('nan'))
                    txt = ""
                    try:
                        valf = float(val)
                        if "ProbSubida" in colname:
                            txt = f"{valf:.2%}" if not pd.isna(valf) else ""
                        else:
                            txt = f"{valf:.2f}" if not pd.isna(valf) else ""
                    except Exception:
                        txt = ""
                    self.top25_table.setItem(i, j, QTableWidgetItem(txt))
                btn = QPushButton("Ver Detalhe")
                ticker = row["Ticker"]
                btn.clicked.connect(lambda _, t=ticker: self.mostrar_detalhe_acao(t))
                self.top25_table.setCellWidget(i, self.top25_table.columnCount() - 1, btn)
            else:
                for j in range(self.top25_table.columnCount()):
                    self.top25_table.setItem(i, j, QTableWidgetItem(""))

    def atualizar_sugestoes(self):
        universo_nome = self.universo_combo.currentText()
        cache_f = self.cache_file(universo_nome)
        if os.path.exists(cache_f):
            r = QMessageBox.question(self, "Usar Cache?",
                f"Já existe uma análise gravada para '{universo_nome}'.\nQueres usar o resultado guardado (mais rápido)?",
                QMessageBox.Yes | QMessageBox.No)
            if r == QMessageBox.Yes:
                self.mostrar_cache()
                return
        universe = self.get_universe(universo_nome)
        if not universe:
            QMessageBox.warning(self, "Erro", f"Não foi possível carregar o universo '{universo_nome}'.")
            self.suggestions_table.setRowCount(0)
            return
        portfolio_tickers = {pos['ticker'] for pos in self.portfolio_manager.positions}
        tickers_para_analise = [t for t in universe if t not in portfolio_tickers]
        self.suggestions_table.setRowCount(len(tickers_para_analise))
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.worker = SuggestionWorker(
            tickers_para_analise, self.data_provider, self.predictor, self.portfolio_manager)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.row_ready.connect(self.atualiza_linha_tabela)
        self.worker.finished.connect(self.termina_worker)
        self.worker.start()
        self.universe_last_scan[universo_nome] = datetime.datetime.now()
        self.update_last_scan_label()

    def atualiza_linha_tabela(self, i, linha):
        self.suggestions_table.setItem(i, 0, QTableWidgetItem(str(linha["Ticker"])))
        self.suggestions_table.setItem(i, 1, QTableWidgetItem(f'{linha.get("PrecoAtual", float("nan")):.2f}' if not pd.isna(linha.get("PrecoAtual", float('nan'))) else "N/A"))
        col = 2
        for m in ["Logistic", "RF", "MLP"]:
            for h in [1, 3]:
                proba = linha.get(f"ProbSubida_{m}_{h}d", float('nan'))
                try:
                    txt = f"{proba:.2%}" if not pd.isna(proba) else ""
                except Exception:
                    txt = ""
                self.suggestions_table.setItem(i, col, QTableWidgetItem(txt)); col += 1
                sug = linha.get(f"Sugestao_{m}_{h}d", "")
                self.suggestions_table.setItem(i, col, QTableWidgetItem(str(sug))); col += 1
        for h in [1, 3]:
            val = linha.get(f"Consenso_ProbSubida_{h}d", float('nan'))
            try:
                txt = f"{val:.2%}" if not pd.isna(val) else ""
            except Exception:
                txt = ""
            self.suggestions_table.setItem(i, col, QTableWidgetItem(txt)); col += 1
        btn = QPushButton("Ver Detalhe")
        btn.clicked.connect(lambda _, t=linha["Ticker"]: self.mostrar_detalhe_acao(t))
        self.suggestions_table.setCellWidget(i, col, btn)

    def termina_worker(self, linhas_cache):
        self.progress_bar.setVisible(False)
        universo_nome = self.universo_combo.currentText()
        cache_f = self.cache_file(universo_nome)
        dfcache = pd.DataFrame(linhas_cache)
        dfcache.to_csv(cache_f, index=False)
        self.preencher_tabela(dfcache)
        QMessageBox.information(self, "Sugestões Atualizadas", f"Análise de {universo_nome} terminada e guardada em cache.")

    def mostrar_detalhe_acao(self, ticker):
        universo_nome = self.universo_combo.currentText()
        cache_f = self.cache_file(universo_nome)
        if not os.path.exists(cache_f):
            QMessageBox.warning(self, "Detalhe", "Sem cache para análise.")
            return
        df = pd.read_csv(cache_f)
        linha = df[df["Ticker"] == ticker]
        if linha.empty:
            QMessageBox.warning(self, "Detalhe", "Ticker não encontrado na cache.")
            return

        preco_atual = linha.iloc[0].get("PrecoAtual", float("nan"))

        results_dict = {}
        for model in ["Logistic", "RF", "MLP"]:
            results_dict[model] = {}
            for n_ahead in [1, 3]:
                prob = linha.iloc[0].get(f"ProbSubida_{model}_{n_ahead}d", float('nan'))
                proba = [1 - prob, prob] if not pd.isna(prob) else [float('nan'), float('nan')]
                results_dict[model][f"proba_{n_ahead}d"] = proba
                results_dict[model][f"price_pred_{n_ahead}d"] = linha.iloc[0].get(f"PrecoPrev_{model}_{n_ahead}d", float('nan'))
                results_dict[model][f"sugestao_{n_ahead}d"] = linha.iloc[0].get(f"Sugestao_{model}_{n_ahead}d", "")
                results_dict[model][f"features_{n_ahead}d"] = linha.iloc[0].get(f"Features_{model}_{n_ahead}d", "")
            results_dict[model]["preco_atual"] = preco_atual

        dlg = SuggestionDetailDialog(ticker, results_dict, self)
        dlg.exec_()

    def exportar_cache(self):
        universo_nome = self.universo_combo.currentText()
        cache_f = self.cache_file(universo_nome)
        if not os.path.exists(cache_f):
            QMessageBox.warning(self, "Exportar", "Não existe cache para exportar.")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Exportar cache CSV", f"{universo_nome}.csv", "CSV Files (*.csv)")
        if file_name:
            try:
                pd.read_csv(cache_f).to_csv(file_name, index=False)
                QMessageBox.information(self, "Exportar", f"Cache exportada para {file_name}.")
            except Exception as e:
                QMessageBox.critical(self, "Erro exportar", str(e))

    def importar_cache(self):
        universo_nome = self.universo_combo.currentText()
        file_name, _ = QFileDialog.getOpenFileName(self, "Importar cache CSV", "", "CSV Files (*.csv)")
        if file_name:
            try:
                df = pd.read_csv(file_name)
                df.to_csv(self.cache_file(universo_nome), index=False)
                self.mostrar_cache()
                QMessageBox.information(self, "Importar", f"Cache importada com sucesso para {universo_nome}.")
            except Exception as e:
                QMessageBox.critical(self, "Erro importar", str(e))
