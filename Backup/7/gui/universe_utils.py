import pandas as pd
import os

def carregar_sp500(cache=True):
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    cache_file = "cache_sp500.csv"
    if cache and os.path.isfile(cache_file):
        try:
            return sorted(list(pd.read_csv(cache_file)["Symbol"].unique()))
        except Exception: pass
    try:
        tabela = pd.read_csv(url)
        tabela.to_csv(cache_file, index=False)
        return sorted(list(tabela["Symbol"].unique()))
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar o S&P500 online: {e}")
        return []

def carregar_nasdaq100_old(cache=True):
    url = "https://raw.githubusercontent.com/eddieoz/NASDAQ-100/main/nasdaq100_list.csv"
    cache_file = "cache_nasdaq100.csv"
    if cache and os.path.isfile(cache_file):
        try:
            return sorted(list(pd.read_csv(cache_file)["Symbol"].unique()))
        except Exception: pass
    try:
        tabela = pd.read_csv(url)
        tabela[["Symbol"]].to_csv(cache_file, index=False)
        return sorted(list(tabela["Symbol"].unique()))
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar o NASDAQ-100 online: {e}")
        return []
    
def carregar_nasdaq100(cache=True):
    ficheiro_local = "nasdaq100_tickers.txt"
    if os.path.isfile(ficheiro_local):
        try:
            with open(ficheiro_local) as f:
                return sorted([linha.strip().upper() for linha in f if linha.strip()])
        except Exception as e:
            print(f"[ERRO] Falha ao carregar tickers do ficheiro local: {e}")
    # Fallback: tenta carregar do cache, se existir
    cache_file = "cache_nasdaq100.csv"
    if os.path.isfile(cache_file):
        try:
            return sorted(list(pd.read_csv(cache_file)["Ticker"].unique()))
        except Exception: pass
    print("[ERRO] Não foi possível carregar a lista do NASDAQ-100 nem do ficheiro local nem do cache.")
    return []

def carregar_psi20(cache=True):
    url = "https://en.wikipedia.org/wiki/PSI-20"
    cache_file = "cache_psi20.csv"
    if cache and os.path.isfile(cache_file):
        try:
            return sorted(list(pd.read_csv(cache_file)["Ticker"].unique()))
        except Exception: pass
    try:
        dfs = pd.read_html(url, header=0)
        tabela = next(df for df in dfs if any("Ticker" in col or "Símbolo" in col for col in df.columns))
        col = [c for c in tabela.columns if "Ticker" in c or "Símbolo" in c][0]
        pd.DataFrame({ "Ticker": tabela[col] }).to_csv(cache_file, index=False)
        return sorted([t.split()[0].replace('.', '-').upper() for t in tabela[col].unique()])
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar o PSI20 online: {e}")
        return []

def carregar_euronext100(cache=True):
    ficheiro_local = "euronext100_tickers.txt"
    if os.path.isfile(ficheiro_local):
        try:
            with open(ficheiro_local, "r") as f:
                return sorted([linha.strip().upper() for linha in f if linha.strip()])
        except Exception as e:
            print(f"[ERRO] Falha ao ler ficheiro Euronext100: {e}")
    # Fallback: método antigo (internet)
    url = "https://en.wikipedia.org/wiki/Euronext_100"
    cache_file = "cache_euronext100.csv"
    if cache and os.path.isfile(cache_file):
        try:
            return sorted(list(pd.read_csv(cache_file)["Ticker"].unique()))
        except Exception: pass
    try:
        dfs = pd.read_html(url, header=0)
        tabela = next(df for df in dfs if any("Ticker" in col or "Symbol" in col for col in df.columns))
        col = [c for c in tabela.columns if "Ticker" in c or "Symbol" in c][0]
        pd.DataFrame({ "Ticker": tabela[col] }).to_csv(cache_file, index=False)
        return sorted([t.split()[0].replace('.', '-').upper() for t in tabela[col].unique()])
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar o Euronext 100 online: {e}")
        return []

def carregar_eurostoxx50(cache=True):
    url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
    cache_file = "cache_eurostoxx50.csv"
    if cache and os.path.isfile(cache_file):
        try:
            return sorted(list(pd.read_csv(cache_file)["Ticker"].unique()))
        except Exception: pass
    try:
        dfs = pd.read_html(url, header=0)
        tabela = next(df for df in dfs if any("Ticker" in col or "Symbol" in col for col in df.columns))
        col = [c for c in tabela.columns if "Ticker" in c or "Symbol" in c][0]
        pd.DataFrame({ "Ticker": tabela[col] }).to_csv(cache_file, index=False)
        return sorted([t.split()[0].replace('.', '-').upper() for t in tabela[col].unique()])
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar o Euro Stoxx 50 online: {e}")
        return []

def carregar_nyse(cache=True):
    import os
    ficheiro = "nyse_tickers.txt"  # Caminho para o ficheiro local
    print("[DEBUG] Caminho absoluto NYSE:", os.path.abspath(ficheiro))  # <--- Adicionado!
    try:
        with open(ficheiro, "r") as f:
            return [linha.strip().upper() for linha in f if linha.strip()]
    except Exception as e:
        print(f"[ERRO] Falha ao carregar ficheiro NYSE: {e}")
        return []

def carregar_tickers_ficheiro(ficheiro="tickers_custom.txt"):
    try:
        with open(ficheiro, "r") as f:
            return [linha.strip().upper() for linha in f if linha.strip()]
    except Exception as e:
        print(f"[ERRO] Falha ao carregar ficheiro '{ficheiro}': {e}")
        return []

def guardar_tickers_ficheiro(tickers, ficheiro="tickers_custom.txt"):
    try:
        with open(ficheiro, "w") as f:
            for t in tickers:
                f.write(f"{t}\n")
    except Exception as e:
        print(f"[ERRO] Falha ao guardar ficheiro '{ficheiro}': {e}")

UNIVERSE_FUNCS = {
    "S&P500": carregar_sp500,
    "NASDAQ100": carregar_nasdaq100,
    "PSI20": carregar_psi20,
    "Euronext100": carregar_euronext100,
    "EuroStoxx50": carregar_eurostoxx50,
    "NYSE": carregar_nyse,
    "Personalizado": carregar_tickers_ficheiro,
}
