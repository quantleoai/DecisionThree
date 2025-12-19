"""
Gravitational Center Indicator (Centro Gravitacional)

Indicador que calcula o ponto médio dinâmico de equilíbrio do preço,
considerando momentum e volatilidade, não apenas média simples.

Autor: Hydra Lab
Versão: 1.0
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from base_indicators import (
    calculate_lwma,
    calculate_ema,
    calculate_atr,
    calculate_momentum,
    calculate_typical_price
)


def calculate_gravitational_center(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 100,
    smooth_period: int = 20,
    smooth_method: str = 'lwma'
) -> pd.Series:
    """
    Calcula o Centro Gravitacional (CG)

    O CG é calculado como média ponderada dos preços, onde o peso
    é dado pela "massa gravitacional" = Volume × Momentum

    Fórmula:
        CG_t = Σ(P_i × M_i) / Σ(M_i)

        onde:
        P_i = Preço típico (High + Low + Close) / 3
        M_i = Massa = Volume_i × Momentum_i
        Momentum_i = |Close_i - Close_(i-1)|

    Args:
        high: Série de máximas
        low: Série de mínimas
        close: Série de fechamento
        volume: Série de volume
        period: Janela para cálculo do CG (default: 100)
        smooth_period: Período de suavização (default: 20)
        smooth_method: Método de suavização ('lwma', 'ema', 'sma')

    Returns:
        Série com Centro Gravitacional suavizado
    """

    # 1. Calcula preço típico (HLC/3)
    typical_price = calculate_typical_price(high, low, close)

    # 2. Calcula momentum (movimento absoluto)
    momentum = calculate_momentum(close, period=1)

    # 3. Calcula massa gravitacional = Volume × Momentum
    # Adiciona pequeno valor para evitar divisão por zero
    mass = volume * (momentum + 1e-10)

    # 4. Calcula CG bruto usando rolling window
    # Usa abordagem vetorizada mais eficiente

    # Cria arrays para cálculo
    price_array = typical_price.values
    mass_array = mass.values

    # Inicializa array de resultados
    cg_raw = np.full(len(price_array), np.nan)

    # Calcula CG para cada janela
    for i in range(period - 1, len(price_array)):
        # Pega janela de dados
        window_prices = price_array[i - period + 1:i + 1]
        window_masses = mass_array[i - period + 1:i + 1]

        # Calcula CG = média ponderada
        total_mass = window_masses.sum()
        if total_mass > 0:
            cg_raw[i] = (window_prices * window_masses).sum() / total_mass

    # Converte para Series
    cg_raw = pd.Series(cg_raw, index=typical_price.index)

    # 5. Suaviza o CG
    if smooth_method == 'lwma':
        cg_smooth = calculate_lwma(cg_raw, smooth_period)
    elif smooth_method == 'ema':
        cg_smooth = calculate_ema(cg_raw, smooth_period)
    else:  # sma
        cg_smooth = cg_raw.rolling(window=smooth_period).mean()

    return cg_smooth


def calculate_cg_deviation(
    close: pd.Series,
    cg: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr_period: int = 100
) -> pd.Series:
    """
    Calcula o desvio do preço em relação ao Centro Gravitacional

    O desvio é normalizado pelo ATR para ter uma medida em
    "unidades de volatilidade"

    Fórmula:
        Desvio_t = (Close_t - CG_t) / ATR_t

    Interpretação:
        > +1: Preço muito acima do centro (possível reversão para baixo)
        0: Preço no centro gravitacional
        < -1: Preço muito abaixo do centro (possível reversão para cima)

    Args:
        close: Série de fechamento
        cg: Série com Centro Gravitacional
        high: Série de máximas
        low: Série de mínimas
        atr_period: Período para cálculo do ATR

    Returns:
        Série com desvio normalizado
    """

    # Calcula ATR
    atr = calculate_atr(high, low, close, period=atr_period)

    # Evita divisão por zero
    atr = atr.replace(0, np.nan)

    # Calcula desvio normalizado
    deviation = (close - cg) / atr

    return deviation


def calculate_cg_bands(
    cg: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 100,
    std_multiplier: float = 1.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Calcula bandas em torno do Centro Gravitacional

    As bandas são posicionadas a ±N × ATR do centro,
    criando um "campo gravitacional"

    Args:
        cg: Série com Centro Gravitacional
        high: Série de máximas
        low: Série de mínimas
        close: Série de fechamento
        atr_period: Período para cálculo do ATR
        std_multiplier: Multiplicador do ATR (default: 1.0)

    Returns:
        Tupla (upper_band, lower_band)
    """

    # Calcula ATR
    atr = calculate_atr(high, low, close, period=atr_period)

    # Calcula bandas
    upper_band = cg + (atr * std_multiplier)
    lower_band = cg - (atr * std_multiplier)

    return upper_band, lower_band


def calculate_cg_full(
    df: pd.DataFrame,
    period: int = 100,
    smooth_period: int = 20,
    atr_period: int = 100,
    band_multiplier: float = 1.0,
    smooth_method: str = 'lwma'
) -> Dict[str, pd.Series]:
    """
    Calcula indicador completo de Centro Gravitacional

    Retorna CG, bandas, desvio e sinais de reversão

    Args:
        df: DataFrame com colunas ['high', 'low', 'close', 'volume']
        period: Janela para cálculo do CG
        smooth_period: Período de suavização
        atr_period: Período para ATR
        band_multiplier: Multiplicador das bandas
        smooth_method: Método de suavização ('lwma', 'ema', 'sma')

    Returns:
        Dicionário com:
            - 'cg': Centro Gravitacional
            - 'upper_band': Banda superior
            - 'lower_band': Banda inferior
            - 'deviation': Desvio normalizado
            - 'overbought': Sinal de sobrecompra (deviation > 1)
            - 'oversold': Sinal de sobrevenda (deviation < -1)
    """

    # Valida colunas necessárias
    required_cols = ['High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame deve conter coluna '{col}'")

    # Extrai séries
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']

    # 1. Calcula Centro Gravitacional
    cg = calculate_gravitational_center(
        high, low, close, volume,
        period=period,
        smooth_period=smooth_period,
        smooth_method=smooth_method
    )

    # 2. Calcula bandas
    upper_band, lower_band = calculate_cg_bands(
        cg, high, low, close,
        atr_period=atr_period,
        std_multiplier=band_multiplier
    )

    # 3. Calcula desvio
    deviation = calculate_cg_deviation(
        close, cg, high, low,
        atr_period=atr_period
    )

    # 4. Gera sinais de reversão
    overbought = deviation > band_multiplier  # Preço muito acima
    oversold = deviation < -band_multiplier   # Preço muito abaixo

    return {
        'cg': cg,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'deviation': deviation,
        'overbought': overbought,
        'oversold': oversold
    }


def get_cg_signal(deviation: float, threshold: float = 1.0) -> str:
    """
    Interpreta o desvio e retorna sinal de trading

    Args:
        deviation: Valor do desvio normalizado
        threshold: Limiar para sinais (default: 1.0)

    Returns:
        String com sinal: 'STRONG_SELL', 'SELL', 'NEUTRAL', 'BUY', 'STRONG_BUY'
    """

    if pd.isna(deviation):
        return 'NEUTRAL'

    if deviation > threshold * 1.5:
        return 'STRONG_SELL'  # Muito longe acima do centro
    elif deviation > threshold:
        return 'SELL'  # Acima do centro
    elif deviation < -threshold * 1.5:
        return 'STRONG_BUY'  # Muito longe abaixo do centro
    elif deviation < -threshold:
        return 'BUY'  # Abaixo do centro
    else:
        return 'NEUTRAL'  # Próximo ao centro
