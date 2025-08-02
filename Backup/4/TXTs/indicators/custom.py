def custom_indicator(series):
    """Exemplo: retorna série normalizada entre 0 e 1."""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return series * 0
    return (series - min_val) / (max_val - min_val)

