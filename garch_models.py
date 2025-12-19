"""
Modelos GARCH de Volatilidade Condicional
GARCH, TGARCH (GJR-GARCH), EGARCH
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calcula retornos logarítmicos

    Args:
        prices: Série de preços

    Returns:
        Retornos logarítmicos
    """
    return np.log(prices / prices.shift(1))


def fit_garch(
    prices: pd.Series,
    p: int = 1,
    q: int = 1,
    vol_model: str = 'GARCH',
    o: int = 0
) -> Optional[Dict]:
    """
    Ajusta modelo GARCH aos preços (EXATAMENTE como exemplo MT5)

    Args:
        prices: Série de preços (close)
        p: Ordem do termo ARCH
        q: Ordem do termo GARCH
        vol_model: Modelo de volatilidade ('GARCH', 'EGARCH')
        o: Ordem do termo assimétrico (para TGARCH, usar vol_model='GARCH' com o=1)

    Returns:
        Dicionário com modelo ajustado e previsões, ou None se falhar
    """
    try:
        from arch import arch_model

        # Remove NaN dos preços
        prices_clean = prices.dropna()

        if len(prices_clean) < 100:
            warnings.warn("Poucos dados para ajustar GARCH (< 100 observações)")
            return None

        # Debug: mostra estatísticas dos preços (desabilitado em produção)
        # print(f"[DEBUG] Preços: min={prices_clean.min():.5f}, max={prices_clean.max():.5f}, mean={prices_clean.mean():.5f}, len={len(prices_clean)}")

        # Cria modelo EXATAMENTE como no exemplo
        # arch_model(data, vol='Garch', p=1, q=1)
        model = arch_model(
            prices_clean,
            vol=vol_model,
            p=p,
            o=o,
            q=q
        )

        # Ajusta com supressão de warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp='off')

        # Extrai volatilidade condicional
        conditional_vol = result.conditional_volatility

        # Debug: mostra estatísticas da volatilidade (desabilitado em produção)
        # print(f"[DEBUG] Volatilidade {vol_model}: min={conditional_vol.min():.6f}, max={conditional_vol.max():.6f}, mean={conditional_vol.mean():.6f}")

        # Cria Series com resultado completo no índice original
        volatility_full = pd.Series(index=prices.index, dtype=float)
        volatility_full.loc[prices_clean.index] = conditional_vol.values

        return {
            'model': result,
            'volatility': volatility_full,
            'params': result.params,
            'aic': result.aic,
            'bic': result.bic,
            'log_likelihood': result.loglikelihood
        }

    except ImportError:
        warnings.warn("Biblioteca 'arch' não instalada. Instale com: pip install arch")
        return None
    except Exception as e:
        warnings.warn(f"Erro ao ajustar {vol_model}: {e}")
        return None


def calculate_garch(
    df: pd.DataFrame,
    price_col: str = 'close',
    p: int = 1,
    q: int = 1
) -> pd.DataFrame:
    """
    Calcula GARCH(p,q) e adiciona ao DataFrame

    Args:
        df: DataFrame com dados OHLCV
        price_col: Coluna de preços a usar
        p: Ordem ARCH
        q: Ordem GARCH

    Returns:
        DataFrame com colunas adicionais de GARCH
    """
    result = df.copy()

    # Ajusta GARCH diretamente com preços (igual ao exemplo: vol='Garch')
    garch_result = fit_garch(df[price_col], p=p, q=q, vol_model='Garch', o=0)

    if garch_result is not None:
        result['garch_volatility'] = garch_result['volatility']

        # Bandas de volatilidade (preço ± 2*volatilidade)
        result['garch_upper'] = df[price_col] + 2 * garch_result['volatility']
        result['garch_lower'] = df[price_col] - 2 * garch_result['volatility']
    else:
        result['garch_volatility'] = np.nan
        result['garch_upper'] = np.nan
        result['garch_lower'] = np.nan

    return result


def calculate_tgarch(
    df: pd.DataFrame,
    price_col: str = 'close',
    p: int = 1,
    o: int = 1,
    q: int = 1
) -> pd.DataFrame:
    """
    Calcula TGARCH (GJR-GARCH) e adiciona ao DataFrame

    TGARCH captura efeitos assimétricos (más notícias aumentam volatilidade mais que boas)

    Args:
        df: DataFrame com dados OHLCV
        price_col: Coluna de preços a usar
        p: Ordem ARCH
        o: Ordem do termo assimétrico
        q: Ordem GARCH

    Returns:
        DataFrame com colunas adicionais de TGARCH
    """
    result = df.copy()

    # Ajusta TGARCH diretamente com preços (igual ao exemplo: vol='GARCH' com o=1)
    tgarch_result = fit_garch(df[price_col], p=p, q=q, vol_model='Garch', o=o)

    if tgarch_result is not None:
        result['tgarch_volatility'] = tgarch_result['volatility']
        result['tgarch_upper'] = df[price_col] + 2 * tgarch_result['volatility']
        result['tgarch_lower'] = df[price_col] - 2 * tgarch_result['volatility']
    else:
        result['tgarch_volatility'] = np.nan
        result['tgarch_upper'] = np.nan
        result['tgarch_lower'] = np.nan

    return result


def calculate_egarch(
    df: pd.DataFrame,
    price_col: str = 'close',
    p: int = 1,
    q: int = 1
) -> pd.DataFrame:
    """
    Calcula EGARCH (Exponential GARCH) e adiciona ao DataFrame

    EGARCH modela log(volatilidade), garantindo volatilidade sempre positiva
    e capturando assimetrias

    Args:
        df: DataFrame com dados OHLCV
        price_col: Coluna de preços a usar
        p: Ordem ARCH
        q: Ordem GARCH

    Returns:
        DataFrame com colunas adicionais de EGARCH
    """
    result = df.copy()

    # Ajusta EGARCH diretamente com preços (igual ao exemplo: vol='EGarch')
    egarch_result = fit_garch(df[price_col], p=p, q=q, vol_model='EGarch', o=0)

    if egarch_result is not None:
        result['egarch_volatility'] = egarch_result['volatility']
        result['egarch_upper'] = df[price_col] + 2 * egarch_result['volatility']
        result['egarch_lower'] = df[price_col] - 2 * egarch_result['volatility']
    else:
        result['egarch_volatility'] = np.nan
        result['egarch_upper'] = np.nan
        result['egarch_lower'] = np.nan

    return result


def calculate_all_garch_models(
    df: pd.DataFrame,
    price_col: str = 'close',
    p: int = 1,
    q: int = 1
) -> pd.DataFrame:
    """
    Calcula todos os modelos GARCH e compara

    Args:
        df: DataFrame com dados OHLCV
        price_col: Coluna de preços a usar
        p: Ordem ARCH
        q: Ordem GARCH

    Returns:
        DataFrame com todas as volatilidades calculadas
    """
    result = df.copy()

    # GARCH
    print("Ajustando GARCH...")
    garch_res = fit_garch(df[price_col], p=p, q=q, vol_model='Garch', o=0)
    if garch_res:
        result['garch_vol'] = garch_res['volatility']
        result['garch_upper'] = df[price_col] + 2 * garch_res['volatility']
        result['garch_lower'] = df[price_col] - 2 * garch_res['volatility']
        print(f"   AIC: {garch_res['aic']:.2f}, BIC: {garch_res['bic']:.2f}")
    else:
        result['garch_vol'] = np.nan
        result['garch_upper'] = np.nan
        result['garch_lower'] = np.nan

    # TGARCH (GJR-GARCH)
    print("Ajustando TGARCH (GJR-GARCH)...")
    tgarch_res = fit_garch(df[price_col], p=p, q=q, vol_model='Garch', o=1)
    if tgarch_res:
        result['tgarch_vol'] = tgarch_res['volatility']
        result['tgarch_upper'] = df[price_col] + 2 * tgarch_res['volatility']
        result['tgarch_lower'] = df[price_col] - 2 * tgarch_res['volatility']
        print(f"   AIC: {tgarch_res['aic']:.2f}, BIC: {tgarch_res['bic']:.2f}")
    else:
        result['tgarch_vol'] = np.nan
        result['tgarch_upper'] = np.nan
        result['tgarch_lower'] = np.nan

    # EGARCH
    print("Ajustando EGARCH...")
    egarch_res = fit_garch(df[price_col], p=p, q=q, vol_model='EGarch', o=0)
    if egarch_res:
        result['egarch_vol'] = egarch_res['volatility']
        result['egarch_upper'] = df[price_col] + 2 * egarch_res['volatility']
        result['egarch_lower'] = df[price_col] - 2 * egarch_res['volatility']
        print(f"   AIC: {egarch_res['aic']:.2f}, BIC: {egarch_res['bic']:.2f}")
    else:
        result['egarch_vol'] = np.nan
        result['egarch_upper'] = np.nan
        result['egarch_lower'] = np.nan

    return result


def compare_garch_models(
    df: pd.DataFrame,
    price_col: str = 'close',
    p: int = 1,
    q: int = 1
) -> pd.DataFrame:
    """
    Compara modelos GARCH e retorna estatísticas

    Args:
        df: DataFrame com dados OHLCV
        price_col: Coluna de preços
        p: Ordem ARCH
        q: Ordem GARCH

    Returns:
        DataFrame com comparação de modelos
    """
    models_info = []

    # GARCH
    garch_result = fit_garch(df[price_col], p=p, q=q, vol_model='Garch', o=0)
    if garch_result:
        models_info.append({
            'Model': 'GARCH',
            'AIC': garch_result['aic'],
            'BIC': garch_result['bic'],
            'LogLikelihood': garch_result['log_likelihood'],
            'Mean_Volatility': garch_result['volatility'].mean(),
            'Std_Volatility': garch_result['volatility'].std()
        })

    # TGARCH
    tgarch_result = fit_garch(df[price_col], p=p, q=q, vol_model='Garch', o=1)
    if tgarch_result:
        models_info.append({
            'Model': 'TGARCH',
            'AIC': tgarch_result['aic'],
            'BIC': tgarch_result['bic'],
            'LogLikelihood': tgarch_result['log_likelihood'],
            'Mean_Volatility': tgarch_result['volatility'].mean(),
            'Std_Volatility': tgarch_result['volatility'].std()
        })

    # EGARCH
    egarch_result = fit_garch(df[price_col], p=p, q=q, vol_model='EGarch', o=0)
    if egarch_result:
        models_info.append({
            'Model': 'EGARCH',
            'AIC': egarch_result['aic'],
            'BIC': egarch_result['bic'],
            'LogLikelihood': egarch_result['log_likelihood'],
            'Mean_Volatility': egarch_result['volatility'].mean(),
            'Std_Volatility': egarch_result['volatility'].std()
        })

    if models_info:
        comparison_df = pd.DataFrame(models_info)
        comparison_df = comparison_df.sort_values('AIC')  # Melhor modelo = menor AIC
        return comparison_df
    else:
        return pd.DataFrame()


def forecast_volatility(
    prices: pd.Series,
    p: int = 1,
    q: int = 1,
    vol_model: str = 'GARCH',
    o: int = 0,
    horizon: int = 1
) -> Optional[Dict]:
    """
    Prevê volatilidade futura usando GARCH (baseado em log returns)

    Args:
        prices: Série de preços
        p: Ordem ARCH
        q: Ordem GARCH
        vol_model: Modelo ('GARCH', 'EGARCH', 'Garch', 'EGarch')
        o: Ordem assimétrica (para TGARCH)
        horizon: Horizonte de previsão (número de períodos à frente)

    Returns:
        Dicionário com previsões
    """
    try:
        from arch import arch_model

        prices_clean = prices.dropna()

        if len(prices_clean) < 100:
            warnings.warn("Poucos dados para previsão (< 100 observações)")
            return None

        # Calcula log returns (como no exemplo MT5)
        log_returns = np.log(prices_clean / prices_clean.shift(1))

        # Remove inf e NaN
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(log_returns) < 100:
            warnings.warn("Poucos dados após calcular retornos (< 100 observações)")
            return None

        # Cria e ajusta modelo com log returns
        model = arch_model(log_returns, vol=vol_model, p=p, o=o, q=q)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp='off')

        # Faz previsão
        forecast = result.forecast(horizon=horizon)

        # Extrai volatilidade prevista (raiz da variância)
        # forecast.variance[-1:] retorna DataFrame, extrai primeiro valor
        variance_forecast = forecast.variance.iloc[-1, 0]  # Última linha, primeira coluna
        volatility_forecast = np.sqrt(variance_forecast)

        # Calcula em percentual (como no exemplo MT5)
        volatility_pct = volatility_forecast * 100

        # Preço atual
        current_price = prices_clean.iloc[-1]

        # Conversão correta de log returns para pontos de preço
        # Para log returns, a movimentação esperada é:
        # Upper = price × exp(σ), Lower = price × exp(-σ)
        upper_price = current_price * np.exp(volatility_forecast)
        lower_price = current_price * np.exp(-volatility_forecast)

        # Range esperado de movimentação (em pontos)
        expected_range = upper_price - lower_price

        # Volatilidade em pontos (aproximação linear: preço × σ)
        volatility_points = current_price * volatility_forecast

        return {
            'model_name': vol_model,
            'volatility_forecast': volatility_forecast,
            'volatility_pct': volatility_pct,
            'volatility_points': volatility_points,
            'current_price': current_price,
            'upper_price': upper_price,
            'lower_price': lower_price,
            'expected_range': expected_range,
            'horizon': horizon,
            'aic': result.aic,
            'bic': result.bic
        }

    except ImportError:
        warnings.warn("Biblioteca 'arch' não instalada")
        return None
    except Exception as e:
        warnings.warn(f"Erro ao prever volatilidade: {e}")
        return None


def forecast_all_models(
    df: pd.DataFrame,
    price_col: str = 'close',
    p: int = 1,
    q: int = 1,
    horizon: int = 1
) -> pd.DataFrame:
    """
    Prevê volatilidade com todos os 3 modelos e compara

    Args:
        df: DataFrame com dados OHLCV
        price_col: Coluna de preços
        p: Ordem ARCH
        q: Ordem GARCH
        horizon: Horizonte de previsão

    Returns:
        DataFrame com comparação das previsões
    """
    forecasts = []

    # GARCH
    print("Prevendo com GARCH...")
    garch_forecast = forecast_volatility(df[price_col], p=p, q=q, vol_model='Garch', o=0, horizon=horizon)
    if garch_forecast:
        forecasts.append({
            'Model': 'GARCH',
            'Volatility_Pct': garch_forecast['volatility_pct'],
            'Volatility_Points': garch_forecast['volatility_points'],
            'Upper_Price': garch_forecast['upper_price'],
            'Lower_Price': garch_forecast['lower_price'],
            'Expected_Range': garch_forecast['expected_range'],
            'AIC': garch_forecast['aic'],
            'BIC': garch_forecast['bic']
        })

    # TGARCH
    print("Prevendo com TGARCH...")
    tgarch_forecast = forecast_volatility(df[price_col], p=p, q=q, vol_model='Garch', o=1, horizon=horizon)
    if tgarch_forecast:
        forecasts.append({
            'Model': 'TGARCH',
            'Volatility_Pct': tgarch_forecast['volatility_pct'],
            'Volatility_Points': tgarch_forecast['volatility_points'],
            'Upper_Price': tgarch_forecast['upper_price'],
            'Lower_Price': tgarch_forecast['lower_price'],
            'Expected_Range': tgarch_forecast['expected_range'],
            'AIC': tgarch_forecast['aic'],
            'BIC': tgarch_forecast['bic']
        })

    # EGARCH
    print("Prevendo com EGARCH...")
    egarch_forecast = forecast_volatility(df[price_col], p=p, q=q, vol_model='EGarch', o=0, horizon=horizon)
    if egarch_forecast:
        forecasts.append({
            'Model': 'EGARCH',
            'Volatility_Pct': egarch_forecast['volatility_pct'],
            'Volatility_Points': egarch_forecast['volatility_points'],
            'Upper_Price': egarch_forecast['upper_price'],
            'Lower_Price': egarch_forecast['lower_price'],
            'Expected_Range': egarch_forecast['expected_range'],
            'AIC': egarch_forecast['aic'],
            'BIC': egarch_forecast['bic']
        })

    if forecasts:
        forecast_df = pd.DataFrame(forecasts)
        forecast_df = forecast_df.sort_values('AIC')  # Melhor modelo = menor AIC
        return forecast_df
    else:
        return pd.DataFrame()
