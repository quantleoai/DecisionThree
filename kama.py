"""
KAMA - Kaufman's Adaptive Moving Average
=========================================

Média móvel adaptativa que se ajusta à volatilidade do mercado.
- Em mercados laterais: KAMA é mais lento (evita whipsaws)
- Em mercados de tendência: KAMA é mais rápido (captura movimentos)

Fórmula:
--------
1. Efficiency Ratio (ER):
   ER = Change / Volatility
   Change = |Close - Close[n]|
   Volatility = Σ|Close - Close[1]| para os últimos n períodos

2. Smoothing Constant (SC):
   SC = [ER × (fast_SC - slow_SC) + slow_SC]²
   fast_SC = 2/(fast+1)
   slow_SC = 2/(slow+1)

3. KAMA:
   KAMA = KAMA[1] + SC × (Close - KAMA[1])

Interpretação:
--------------
- ER ≈ 1: Mercado em tendência forte
- ER ≈ 0: Mercado lateral/volátil
- KAMA cruza preço para cima: Sinal de compra
- KAMA cruza preço para baixo: Sinal de venda

Autor: Hydra Lab
Versão: 1.0
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict


def calculate_efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Calcula o Efficiency Ratio (ER)

    ER = |Change| / Volatility

    Args:
        close: Série de preços de fechamento
        period: Período para cálculo

    Returns:
        Série com Efficiency Ratio (0 a 1)
    """
    # Change: movimento direcional
    change = abs(close - close.shift(period))

    # Volatility: soma dos movimentos absolutos
    volatility = abs(close - close.shift(1)).rolling(window=period).sum()

    # ER = Change / Volatility
    er = change / volatility

    # Limita entre 0 e 1
    er = er.clip(0, 1)

    return er


def calculate_kama(
    close: pd.Series,
    period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30
) -> pd.Series:
    """
    Calcula KAMA - Kaufman's Adaptive Moving Average

    Args:
        close: Série de preços de fechamento
        period: Período para Efficiency Ratio (default: 10)
        fast_period: Período rápido para SC (default: 2)
        slow_period: Período lento para SC (default: 30)

    Returns:
        Série com valores do KAMA
    """
    # Calcula Efficiency Ratio
    er = calculate_efficiency_ratio(close, period)

    # Calcula Smoothing Constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)

    # Smoothing Constant adaptativo
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Inicializa KAMA
    kama = pd.Series(index=close.index, dtype=float)

    # Primeiro valor válido: usa SMA
    first_valid_idx = close.first_valid_index()
    start_idx = close.index.get_loc(first_valid_idx) + period

    if start_idx < len(close):
        kama.iloc[start_idx] = close.iloc[start_idx]

        # Calcula KAMA recursivamente
        for i in range(start_idx + 1, len(close)):
            if pd.notna(sc.iloc[i]) and pd.notna(kama.iloc[i-1]):
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])

    return kama


def calculate_kama_bands(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    kama: pd.Series,
    atr_period: int = 14,
    band_mult: float = 1.5
) -> Tuple[pd.Series, pd.Series]:
    """
    Calcula bandas em torno do KAMA baseadas no ATR

    Args:
        close: Série de fechamento
        high: Série de máximas
        low: Série de mínimas
        kama: Série do KAMA
        atr_period: Período do ATR
        band_mult: Multiplicador das bandas

    Returns:
        Tupla (upper_band, lower_band)
    """
    # Calcula ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    # Bandas
    upper = kama + (atr * band_mult)
    lower = kama - (atr * band_mult)

    return upper, lower


def calculate_kama_signal(close: pd.Series, kama: pd.Series) -> pd.Series:
    """
    Gera sinais de trading baseados no KAMA

    Args:
        close: Série de fechamento
        kama: Série do KAMA

    Returns:
        Série com sinais: 1 (bullish), -1 (bearish), 0 (neutral)
    """
    signal = pd.Series(index=close.index, data=0)

    # Bullish: preço acima do KAMA
    signal[close > kama] = 1

    # Bearish: preço abaixo do KAMA
    signal[close < kama] = -1

    return signal


def calculate_kama_crossover(close: pd.Series, kama: pd.Series) -> pd.Series:
    """
    Detecta crossovers entre preço e KAMA

    Args:
        close: Série de fechamento
        kama: Série do KAMA

    Returns:
        Série com crossovers: 1 (cruz para cima), -1 (cruz para baixo), 0 (sem cruzamento)
    """
    position = (close > kama).astype(int)
    crossover = position.diff()

    return crossover


def calculate_kama_full(
    df: pd.DataFrame,
    period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
    atr_period: int = 14,
    band_mult: float = 1.5
) -> Dict[str, pd.Series]:
    """
    Calcula KAMA completo com bandas, ER e sinais

    Args:
        df: DataFrame com colunas ['high', 'low', 'close']
        period: Período para ER
        fast_period: Período rápido
        slow_period: Período lento
        atr_period: Período ATR para bandas
        band_mult: Multiplicador das bandas

    Returns:
        Dicionário com:
        - 'kama': Linha principal do KAMA
        - 'upper_band': Banda superior
        - 'lower_band': Banda inferior
        - 'efficiency_ratio': ER (0-1)
        - 'signal': Sinal de trading
        - 'crossover': Crossovers detectados
    """
    close = df['close']
    high = df['high']
    low = df['low']

    # KAMA principal
    kama = calculate_kama(close, period, fast_period, slow_period)

    # Efficiency Ratio
    er = calculate_efficiency_ratio(close, period)

    # Bandas
    upper, lower = calculate_kama_bands(close, high, low, kama, atr_period, band_mult)

    # Sinais
    signal = calculate_kama_signal(close, kama)
    crossover = calculate_kama_crossover(close, kama)

    return {
        'kama': kama,
        'upper_band': upper,
        'lower_band': lower,
        'efficiency_ratio': er,
        'signal': signal,
        'crossover': crossover
    }


def get_kama_interpretation(er: float, signal: int) -> str:
    """
    Interpreta os valores do KAMA

    Args:
        er: Efficiency Ratio atual
        signal: Sinal atual (1, -1, 0)

    Returns:
        String com interpretação
    """
    # Interpretação do ER
    if er > 0.7:
        market = "TENDÊNCIA FORTE"
    elif er > 0.4:
        market = "TENDÊNCIA MODERADA"
    else:
        market = "MERCADO LATERAL"

    # Interpretação do sinal
    if signal == 1:
        direction = "BULLISH"
    elif signal == -1:
        direction = "BEARISH"
    else:
        direction = "NEUTRO"

    return f"{direction} | {market} (ER: {er:.2f})"


if __name__ == "__main__":
    """Teste do indicador KAMA"""
    import yfinance as yf

    print("Testando KAMA com AAPL...")

    # Baixa dados
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="6mo")
    df.columns = df.columns.str.lower()

    # Calcula KAMA
    result = calculate_kama_full(df)

    # Últimos valores
    last_idx = -1
    print(f"\nÚltimos valores:")
    print(f"  Close: ${df['close'].iloc[last_idx]:.2f}")
    print(f"  KAMA: ${result['kama'].iloc[last_idx]:.2f}")
    print(f"  Upper Band: ${result['upper_band'].iloc[last_idx]:.2f}")
    print(f"  Lower Band: ${result['lower_band'].iloc[last_idx]:.2f}")
    print(f"  Efficiency Ratio: {result['efficiency_ratio'].iloc[last_idx]:.3f}")
    print(f"  Signal: {result['signal'].iloc[last_idx]}")

    interpretation = get_kama_interpretation(
        result['efficiency_ratio'].iloc[last_idx],
        result['signal'].iloc[last_idx]
    )
    print(f"\n  Interpretação: {interpretation}")
