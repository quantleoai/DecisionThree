"""
Hurst Exponent Indicator
========================

Calcula o Expoente de Hurst (H) para detectar persistência/anti-persistência em séries temporais.

Definição Matemática:
--------------------
O Expoente de Hurst descreve a relação entre amplitude acumulada (R) e desvio padrão (S):

E[R(n) / S(n)] = C * n^H   (quando n → ∞)

Componentes:
-----------
1. Média da série:
   X_bar = (1/N) * Σ(i=1→N) X_i

2. Desvios da média:
   Y_i = X_i - X_bar

3. Série acumulada dos desvios:
   Z_t = Σ(i=1→t) Y_i

4. Range acumulado:
   R(n) = max(Z_1...Z_n) - min(Z_1...Z_n)

5. Desvio padrão:
   S(n) = sqrt( (1/N) * Σ(i=1→N) (X_i - X_bar)^2 )

6. Relação logarítmica:
   log(R/S) = log(C) + H * log(n)
   → H é o coeficiente angular da regressão linear

Interpretação:
-------------
H < 0.5  → Anti-persistente (reversão à média)
H = 0.5  → Ruído branco (random walk)
H > 0.5  → Persistente (tendência)

Uso em Trading:
--------------
- H > 0.6: mercado em tendência → estratégias momentum
- H < 0.4: mercado mean-reverting → estratégias de reversão
- H ≈ 0.5: mercado aleatório → evitar trading ou usar ambas

Autor: Hydra Lab
Versão: 1.0.0
Data: 2025-11-09
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from numpy.polynomial.polynomial import polyfit


def calculate_rs(data: np.ndarray) -> Tuple[float, float]:
    """
    Calcula R/S (Range over Standard Deviation) para uma série.

    Args:
        data: Array com os retornos (não preços brutos!)

    Returns:
        Tuple com (R, S) onde:
        - R: Range acumulado
        - S: Desvio padrão
    """
    n = len(data)

    if n < 2:
        return 0.0, 1.0

    # 1. Média da série
    mean = np.mean(data)

    # 2. Desvios da média
    deviations = data - mean

    # 3. Série acumulada dos desvios
    cumsum = np.cumsum(deviations)

    # 4. Range acumulado
    R = np.max(cumsum) - np.min(cumsum)

    # 5. Desvio padrão
    S = np.std(data, ddof=1)  # ddof=1 para desvio padrão amostral

    # Evita divisão por zero
    if S == 0:
        S = 1e-10

    return R, S


def hurst_exponent(
    prices: pd.Series,
    min_window: int = 10,
    max_window: Optional[int] = None,
    num_windows: int = 20
) -> float:
    """
    Calcula o Expoente de Hurst usando o método R/S (Rescaled Range).

    IMPORTANTE: Usa retornos logarítmicos ao invés de preços brutos para
    garantir estacionariedade da série.

    Args:
        prices: Série de preços
        min_window: Tamanho mínimo da janela
        max_window: Tamanho máximo da janela (None = metade do tamanho da série)
        num_windows: Número de janelas para testar

    Returns:
        Expoente de Hurst (H)
    """
    # Converte preços para retornos logarítmicos (série estacionária)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    data = log_returns.values
    n = len(data)

    # Define max_window se não fornecido
    if max_window is None:
        max_window = n // 2

    # Garante valores mínimos
    min_window = max(min_window, 10)
    max_window = min(max_window, n - 1)

    if max_window <= min_window:
        return 0.5  # Retorna random walk se não há dados suficientes

    # Cria array de tamanhos de janela (espaçamento logarítmico)
    window_sizes = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num=num_windows
        ).astype(int)
    )

    # Arrays para armazenar log(n) e log(R/S)
    log_n = []
    log_rs = []

    # Calcula R/S para cada tamanho de janela
    for window in window_sizes:
        if window >= n:
            continue

        # Divide a série em segmentos não sobrepostos
        num_segments = n // window
        rs_values = []

        for i in range(num_segments):
            start_idx = i * window
            end_idx = start_idx + window
            segment = data[start_idx:end_idx]

            R, S = calculate_rs(segment)

            if S > 0:
                rs_values.append(R / S)

        if len(rs_values) > 0:
            # Média de R/S para este tamanho de janela
            mean_rs = np.mean(rs_values)

            if mean_rs > 0:
                log_n.append(np.log10(window))
                log_rs.append(np.log10(mean_rs))

    # Regressão linear: log(R/S) = log(C) + H * log(n)
    # H é o coeficiente angular (slope)
    if len(log_n) < 2:
        return 0.5  # Não há dados suficientes

    # polyfit retorna [c0, c1, ...] onde y = c0 + c1*x + c2*x^2 + ...
    # Para grau 1: y = c0 + c1*x, então slope = c1
    coeffs = polyfit(log_n, log_rs, 1)
    H = coeffs[1]  # slope = c1

    # Limita H ao intervalo [0, 1]
    H = np.clip(H, 0.0, 1.0)

    return H


def hurst_rolling(
    prices: pd.Series,
    period: int = 100,
    min_window: int = 10,
    max_window: Optional[int] = None,
    num_windows: int = 20
) -> pd.Series:
    """
    Calcula o Expoente de Hurst rolante (rolling window).

    Args:
        prices: Série de preços
        period: Período da janela rolante
        min_window: Tamanho mínimo da janela para cálculo de R/S
        max_window: Tamanho máximo da janela para cálculo de R/S
        num_windows: Número de janelas para testar

    Returns:
        Série com valores de H
    """
    result = pd.Series(index=prices.index, dtype=float)

    for i in range(period, len(prices) + 1):
        window_data = prices.iloc[i - period:i]

        H = hurst_exponent(
            window_data,
            min_window=min_window,
            max_window=max_window,
            num_windows=num_windows
        )

        result.iloc[i - 1] = H

    return result


def hurst_regime(H: float) -> str:
    """
    Classifica o regime de mercado baseado no Expoente de Hurst.

    Args:
        H: Expoente de Hurst

    Returns:
        String com classificação do regime
    """
    if H > 0.6:
        return "Trending"
    elif H < 0.4:
        return "Mean-Reverting"
    else:
        return "Random Walk"


def hurst_signal(H: float, threshold_trending: float = 0.6, threshold_reverting: float = 0.4) -> int:
    """
    Gera sinal de trading baseado no Expoente de Hurst.

    Args:
        H: Expoente de Hurst
        threshold_trending: Threshold para regime de tendência
        threshold_reverting: Threshold para regime de reversão

    Returns:
        +1: Use estratégias momentum (trending)
        -1: Use estratégias mean-reverting
         0: Mercado aleatório (evite trading)
    """
    if H > threshold_trending:
        return 1  # Trending
    elif H < threshold_reverting:
        return -1  # Mean-reverting
    else:
        return 0  # Random walk


# ============================================================================
# Funções auxiliares para análise
# ============================================================================

def calculate_hurst_statistics(prices: pd.Series, period: int = 100) -> dict:
    """
    Calcula estatísticas do Expoente de Hurst para análise.

    Args:
        prices: Série de preços
        period: Período da janela rolante

    Returns:
        Dict com estatísticas
    """
    hurst_series = hurst_rolling(prices, period=period)

    # Remove NaN
    hurst_clean = hurst_series.dropna()

    if len(hurst_clean) == 0:
        return {
            'mean': 0.5,
            'std': 0.0,
            'min': 0.5,
            'max': 0.5,
            'trending_pct': 0.0,
            'reverting_pct': 0.0,
            'random_pct': 100.0
        }

    stats = {
        'mean': hurst_clean.mean(),
        'std': hurst_clean.std(),
        'min': hurst_clean.min(),
        'max': hurst_clean.max(),
        'trending_pct': (hurst_clean > 0.6).sum() / len(hurst_clean) * 100,
        'reverting_pct': (hurst_clean < 0.4).sum() / len(hurst_clean) * 100,
        'random_pct': ((hurst_clean >= 0.4) & (hurst_clean <= 0.6)).sum() / len(hurst_clean) * 100
    }

    return stats


def hurst_confidence(
    prices: pd.Series,
    num_bootstrap: int = 100,
    sample_frac: float = 0.8
) -> Tuple[float, float, float]:
    """
    Calcula intervalo de confiança do Expoente de Hurst usando bootstrap.

    Args:
        prices: Série de preços
        num_bootstrap: Número de amostras bootstrap
        sample_frac: Fração da série para cada amostra

    Returns:
        Tuple com (H_mean, H_lower, H_upper) - média e intervalo de 95%
    """
    n = len(prices)
    sample_size = int(n * sample_frac)

    hurst_values = []

    for _ in range(num_bootstrap):
        # Amostra aleatória com reposição
        indices = np.random.choice(n, size=sample_size, replace=True)
        sample = prices.iloc[indices].reset_index(drop=True)

        H = hurst_exponent(sample)
        hurst_values.append(H)

    hurst_array = np.array(hurst_values)

    H_mean = np.mean(hurst_array)
    H_lower = np.percentile(hurst_array, 2.5)
    H_upper = np.percentile(hurst_array, 97.5)

    return H_mean, H_lower, H_upper


if __name__ == "__main__":
    """Teste básico do indicador"""

    # Gera dados de teste
    np.random.seed(42)

    # 1. Tendência persistente (H > 0.5)
    # Retornos com autocorrelação positiva
    returns_trend = np.zeros(1000)
    returns_trend[0] = np.random.randn() * 0.01
    for i in range(1, 1000):
        returns_trend[i] = 0.3 * returns_trend[i-1] + np.random.randn() * 0.01
    prices_trend = pd.Series(100 * np.exp(np.cumsum(returns_trend)))

    # 2. Reversão à média (H < 0.5)
    # Retornos com autocorrelação negativa
    returns_rev = np.zeros(1000)
    returns_rev[0] = np.random.randn() * 0.01
    for i in range(1, 1000):
        returns_rev[i] = -0.3 * returns_rev[i-1] + np.random.randn() * 0.01
    prices_rev = pd.Series(100 * np.exp(np.cumsum(returns_rev)))

    # 3. Random walk (H ≈ 0.5)
    # Retornos IID (independentes e identicamente distribuídos)
    returns_random = np.random.randn(1000) * 0.01
    prices_random = pd.Series(100 * np.exp(np.cumsum(returns_random)))

    # Testa cada série
    print("=" * 60)
    print("TESTE DO INDICADOR DE HURST")
    print("=" * 60)

    for name, prices in [("Tendência", prices_trend),
                          ("Mean-Reverting", prices_rev),
                          ("Random Walk", prices_random)]:

        H = hurst_exponent(prices)
        regime = hurst_regime(H)
        signal = hurst_signal(H)

        H_mean, H_lower, H_upper = hurst_confidence(prices, num_bootstrap=50)

        print(f"\n{name}:")
        print(f"  H = {H:.3f}")
        print(f"  Intervalo 95%: [{H_lower:.3f}, {H_upper:.3f}]")
        print(f"  Regime: {regime}")
        print(f"  Sinal: {signal} ({'Momentum' if signal > 0 else 'Reverting' if signal < 0 else 'Neutro'})")

    print("\n" + "=" * 60)
    print("✅ Teste concluído!")
