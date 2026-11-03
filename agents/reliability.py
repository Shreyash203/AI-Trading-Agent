"""Pure rules and citation checks used by both runtime and evaluations."""
import math


def calculate_signal(q, sentiment):
    required = ('current_price', 'rsi_14', '50_day_ma', 'macd', 'macd_signal', 'previous_macd', 'previous_macd_signal', 'bb_lower')
    if any(not isinstance(q.get(k), (int, float)) or not math.isfinite(q[k]) for k in required):
        return None
    if q['current_price'] <= 0 or q['50_day_ma'] <= 0 or q['bb_lower'] < 0 or not 0 <= q['rsi_14'] <= 100:
        return None
    if sentiment is None or not math.isfinite(sentiment) or not -1 <= sentiment <= 1:
        return None
    if q['rsi_14'] > 65 or q['current_price'] < q['50_day_ma'] * .95:
        return 'SELL'
    crossover = q['previous_macd'] <= q['previous_macd_signal'] and q['macd'] > q['macd_signal']
    if q['rsi_14'] < 40 and (crossover or q['current_price'] <= q['bb_lower']) and sentiment >= -.5:
        return 'BUY'
    return 'HOLD'


def validate_citations(ids, evidence):
    if not ids or not set(ids) <= {item['id'] for item in evidence}:
        raise ValueError('Missing or unknown source citation')
    return list(dict.fromkeys(ids))
