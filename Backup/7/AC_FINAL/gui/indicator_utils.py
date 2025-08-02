def analyse_indicators_custom(data):
    from indicators.ta import sma, rsi, macd, bollinger_bands
    close = data['Close']
    sma20 = sma(close, 20)
    rsi14 = rsi(close, 14)
    macd_line, macd_signal = macd(close)
    bb_sma, bb_upper, bb_lower = bollinger_bands(close)
    msgs = []
    try:
        if close.iloc[-1] > sma20.iloc[-1]:
            msgs.append("Preço acima da SMA20 — tendência de subida.")
        else:
            msgs.append("Preço abaixo da SMA20 — tendência de descida.")
        if rsi14.iloc[-1] > 70:
            msgs.append("RSI14 acima de 70 — sobrecomprado (pode descer).")
        elif rsi14.iloc[-1] < 30:
            msgs.append("RSI14 abaixo de 30 — sobrevendido (pode subir).")
        else:
            msgs.append("RSI14 em zona neutra.")
        if macd_line.iloc[-1] > macd_signal.iloc[-1]:
            msgs.append("MACD acima do sinal — momentum positivo.")
        else:
            msgs.append("MACD abaixo do sinal — momentum negativo.")
        if close.iloc[-1] >= bb_upper.iloc[-1]:
            msgs.append("Preço tocou na banda superior de Bollinger — potencial reversão para baixo.")
        elif close.iloc[-1] <= bb_lower.iloc[-1]:
            msgs.append("Preço tocou na banda inferior de Bollinger — potencial reversão para cima.")
    except Exception as e:
        msgs.append(f"Erro na análise automática: {e}")
    return " | ".join(msgs)

