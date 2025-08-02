
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
                if data is None or data.empty or 'Close' not in data.columns:
                    for m in ["Logistic", "RF", "MLP"]:
                        for h in [1, 3]:
                            res[f"direction_{m}_{h}d"] = None
                            res[f"proba_{m}_{h}d"] = [float('nan'), float('nan')]
                            res[f"price_pred_{m}_{h}d"] = float('nan')
                            res[f"features_{m}_{h}d"] = {}
                            res[f"sugestao_{m}_{h}d"] = ""
                else:
                    for model, m_key in [("logistic", "Logistic"), ("rf", "RF"), ("mlp", "MLP")]:
                        for n_ahead in [1, 3]:
                            try:
                                self.predictor.model_type = model
                                self.predictor.n_ahead = n_ahead
                                self.predictor.train_on_data(data, model_type=model)
                                direction = self.predictor.predict_direction(data)
                                proba = self.predictor.predict_proba(data)
                                price_pred = self.predictor.predict_price(data)
                                features = self.predictor.get_last_features(data)
                                sug = self.gerar_sugestao(proba)
                            except Exception as e:
                                print(f"[ERRO {ticker} {m_key} {n_ahead}d]: {e}")
                                direction = None
                                proba = [float('nan'), float('nan')]
                                price_pred = float('nan')
                                features = {}
                                sug = ""
                            res[f"direction_{m_key}_{n_ahead}d"] = direction
                            res[f"proba_{m_key}_{n_ahead}d"] = proba
                            res[f"price_pred_{m_key}_{n_ahead}d"] = price_pred
                            res[f"features_{m_key}_{n_ahead}d"] = features
                            res[f"sugestao_{m_key}_{n_ahead}d"] = sug
                # Consensos (médias dos modelos, para cada horizonte)
                for n_ahead in [1, 3]:
                    probs = []
                    for m in ["Logistic", "RF", "MLP"]:
                        p = res.get(f"proba_{m}_{n_ahead}d", [float('nan'), float('nan')])
                        if p and len(p) > 1 and not pd.isna(p[1]):
                            probs.append(p[1])
                    res[f"Consenso_ProbSubida_{n_ahead}d"] = float(pd.Series(probs).mean(skipna=True)) if probs else float('nan')
                    precos = [res.get(f"price_pred_{m}_{n_ahead}d", float('nan')) for m in ["Logistic", "RF", "MLP"]]
                    res[f"Consenso_PrecoPrev_{n_ahead}d"] = float(pd.Series(precos).mean(skipna=True)) if precos else float('nan')
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

    def setup_suggestions_table(self):
        headers = [
            "Ticker",
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
            "Ticker", "Consenso ProbSubida 1d", "Consenso PrecoPrev 1d",
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
            # Força conversão para float das colunas que interessam
            for col in df.columns:
                if any(x in col for x in [
                    "ProbSubida", "PrecoPrev", "Consenso_ProbSubida", "Consenso_PrecoPrev"
                ]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            self.preencher_tabela(df)
        else:
            self.suggestions_table.setRowCount(0)
            self.top25_table.setRowCount(0)

    def preencher_tabela(self, df):
        self.suggestions_table.setRowCount(len(df))
        for i, row in df.iterrows():
            self.suggestions_table.setItem(i, 0, QTableWidgetItem(str(row["Ticker"])))
            col = 1
            for m in ["Logistic", "RF", "MLP"]:
                for h in [1, 3]:
                    # Probabilidade
                    val = row.get(f"ProbSubida_{m}_{h}d", float('nan'))
                    try:
                        valf = float(val)
                        txt = f"{valf:.2%}" if not pd.isna(valf) else ""
                    except Exception:
                        txt = ""
                    self.suggestions_table.setItem(i, col, QTableWidgetItem(txt))
                    col += 1
                    # Sugestão
                    sug = row.get(f"Sugestao_{m}_{h}d", "")
                    self.suggestions_table.setItem(i, col, QTableWidgetItem(str(sug)))
                    col += 1
            for h in [1, 3]:
                val = row.get(f"Consenso_ProbSubida_{h}d", float('nan'))
                try:
                    valf = float(val)
                    txt = f"{valf:.2%}" if not pd.isna(valf) else ""
                except Exception:
                    txt = ""
                self.suggestions_table.setItem(i, col, QTableWidgetItem(txt)); col += 1
            for m in ["Logistic", "RF", "MLP"]:
                for h in [1, 3]:
                    val = row.get(f"PrecoPrev_{m}_{h}d", float('nan'))
                    try:
                        valf = float(val)
                        txt = f"{valf:.2f}" if not pd.isna(valf) else ""
                    except Exception:
                        txt = ""
                    self.suggestions_table.setItem(i, col, QTableWidgetItem(txt)); col += 1
            for h in [1, 3]:
                val = row.get(f"Consenso_PrecoPrev_{h}d", float('nan'))
                try:
                    valf = float(val)
                    txt = f"{valf:.2f}" if not pd.isna(valf) else ""
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
        df_top["Consenso_ProbSubida_1d"] = pd.to_numeric(df_top["Consenso_ProbSubida_1d"], errors="coerce")
        df_top = df_top[df_top["Consenso_ProbSubida_1d"].notna() & (df_top["Consenso_ProbSubida_1d"] > 0.001)]
        df_top = df_top.sort_values("Consenso_ProbSubida_1d", ascending=False).head(25)
        self.top25_table.setRowCount(25)
        for i in range(25):
            if i < len(df_top):
                row = df_top.iloc[i]
                self.top25_table.setItem(i, 0, QTableWidgetItem(str(row["Ticker"])))
                for j, colname in enumerate([
                    "Consenso_ProbSubida_1d", "Consenso_PrecoPrev_1d", "Consenso_ProbSubida_3d", "Consenso_PrecoPrev_3d"
                ], start=1):
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
                # Botão "Ver Detalhe" SEMPRE na última coluna!
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
        col = 1
        self.suggestions_table.setItem(i, 0, QTableWidgetItem(str(linha["Ticker"])))
        for m in ["Logistic", "RF", "MLP"]:
            for h in [1, 3]:
                proba = linha.get(f"proba_{m}_{h}d", [float('nan'), float('nan')])
                try:
                    prob_up = proba[1] if proba and len(proba) > 1 else float('nan')
                    txt = f"{prob_up:.2%}" if not pd.isna(prob_up) else ""
                except Exception:
                    txt = ""
                self.suggestions_table.setItem(i, col, QTableWidgetItem(txt)); col += 1
                sug = linha.get(f"sugestao_{m}_{h}d", "")
                self.suggestions_table.setItem(i, col, QTableWidgetItem(str(sug))); col += 1
        for h in [1, 3]:
            val = linha.get(f"Consenso_ProbSubida_{h}d", float('nan'))
            try:
                valf = float(val)
                txt = f"{valf:.2%}" if not pd.isna(valf) else ""
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

        results_dict = {}
        for model in ["Logistic", "RF", "MLP"]:
            results_dict[model] = {}
            for n_ahead in [1, 3]:
                prob = linha.iloc[0].get(f"ProbSubida_{model}_{n_ahead}d", float('nan'))
                # Passa como lista, se possível
                if not pd.isna(prob):
                    proba = [1 - prob, prob]
                else:
                    proba = [float('nan'), float('nan')]
                results_dict[model][f"proba_{n_ahead}d"] = proba
                results_dict[model][f"PrecoPrev_{n_ahead}d"] = linha.iloc[0].get(f"PrecoPrev_{model}_{n_ahead}d", float('nan'))
                results_dict[model][f"Sugestao_{n_ahead}d"] = linha.iloc[0].get(f"Sugestao_{model}_{n_ahead}d", "")
                # (Opcional) se guardaste features, adiciona:
                results_dict[model][f"features_{n_ahead}d"] = linha.iloc[0].get(f"Features_{model}_{n_ahead}d", "")

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
