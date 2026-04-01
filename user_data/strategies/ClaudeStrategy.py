"""
ClaudeStrategy — Freqtrade strategy with Claude AI as strategic layer.

Claude analyses market conditions every hour and adjusts:
- Which pairs to trade (via custom_pair_whitelist)
- Confidence threshold for entries
- Position sizing bias

Set ANTHROPIC_API_KEY in your environment or .env file.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy.interface import ExitCheckTuple
import talib.abstract as ta

logger = logging.getLogger(__name__)


class ClaudeStrategy(IStrategy):
    """
    Base technical strategy (EMA crossover + RSI filter) with
    Claude as an hourly macro overlay that can veto or confirm signals.
    """

    INTERFACE_VERSION = 3

    # --- Risk management ---
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.03
    }
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    timeframe = "5m"
    startup_candle_count = 50

    # --- Claude state (shared across candles) ---
    _claude_client: Optional[anthropic.Anthropic] = None
    _last_claude_check: Optional[datetime] = None
    _claude_bias: str = "neutral"       # "bullish" | "neutral" | "bearish"
    _claude_reasoning: str = ""
    _claude_check_interval = timedelta(hours=1)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not ANTHROPIC_AVAILABLE:
            logger.warning(
                "ClaudeStrategy: anthropic package not installed. "
                "Running without Claude overlay — all signals pass through."
            )
        elif api_key:
            self._claude_client = anthropic.Anthropic(api_key=api_key)
            logger.info("ClaudeStrategy: Anthropic client initialised.")
        else:
            logger.warning(
                "ClaudeStrategy: ANTHROPIC_API_KEY not set. "
                "Running without Claude overlay — all signals pass through."
            )

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_mean"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    # ------------------------------------------------------------------
    # Entry signal
    # ------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._maybe_update_claude_bias(dataframe, metadata["pair"])

        conditions = (
            (dataframe["ema_fast"] > dataframe["ema_slow"]) &
            (dataframe["rsi"] > 45) &
            (dataframe["rsi"] < 70) &
            (dataframe["volume"] > dataframe["volume_mean"])
        )

        # Claude veto: skip entries when bias is bearish
        if self._claude_bias == "bearish":
            logger.info(
                f"[{metadata['pair']}] Claude bias is BEARISH — entry signals suppressed."
            )
            dataframe["enter_long"] = 0
            dataframe["enter_tag"] = ""
            return dataframe

        dataframe.loc[conditions, "enter_long"] = 1
        dataframe.loc[conditions, "enter_tag"] = f"ema_cross|claude_{self._claude_bias}"
        return dataframe

    # ------------------------------------------------------------------
    # Exit signal
    # ------------------------------------------------------------------

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = (
            (dataframe["ema_fast"] < dataframe["ema_slow"]) |
            (dataframe["rsi"] > 75)
        )

        # Claude bearish: exit faster
        if self._claude_bias == "bearish":
            conditions = conditions | (dataframe["rsi"] > 65)

        dataframe.loc[conditions, "exit_long"] = 1
        return dataframe

    # ------------------------------------------------------------------
    # Claude integration
    # ------------------------------------------------------------------

    def _maybe_update_claude_bias(self, dataframe: DataFrame, pair: str) -> None:
        """Call Claude at most once per hour to get a macro bias."""
        now = datetime.utcnow()
        if (
            self._last_claude_check is not None
            and now - self._last_claude_check < self._claude_check_interval
        ):
            return

        if self._claude_client is None:
            return

        try:
            bias, reasoning = self._ask_claude_for_bias(dataframe, pair)
            self._claude_bias = bias
            self._claude_reasoning = reasoning
            self._last_claude_check = now
            logger.info(
                f"Claude bias updated → {bias.upper()} | {reasoning[:120]}"
            )
        except Exception as e:
            logger.error(f"Claude API error: {e}. Keeping previous bias: {self._claude_bias}")

    def _ask_claude_for_bias(self, dataframe: DataFrame, pair: str) -> tuple[str, str]:
        """
        Send recent OHLCV summary to Claude and get a bias + reasoning back.
        Returns ("bullish"|"neutral"|"bearish", reasoning_string).
        """
        recent = dataframe.tail(24).copy()
        price_now = float(recent["close"].iloc[-1])
        price_24h_ago = float(recent["close"].iloc[0])
        pct_change = ((price_now - price_24h_ago) / price_24h_ago) * 100
        rsi_now = float(recent["rsi"].iloc[-1]) if "rsi" in recent.columns else 50.0
        vol_ratio = float(
            recent["volume"].iloc[-1] / recent["volume"].mean()
        ) if recent["volume"].mean() > 0 else 1.0

        prompt = f"""You are a crypto trading risk manager. Analyse the following 2-hour market snapshot and decide the macro bias.

Pair: {pair}
Current price: {price_now:.4f} USDT
2h price change: {pct_change:+.2f}%
RSI (14): {rsi_now:.1f}
Volume vs 2h average: {vol_ratio:.2f}x

Respond with EXACTLY this JSON format (no extra text):
{{
  "bias": "bullish" | "neutral" | "bearish",
  "reasoning": "one sentence max"
}}"""

        message = self._claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        text = message.content[0].text.strip()
        data = json.loads(text)
        bias = data.get("bias", "neutral").lower()
        if bias not in ("bullish", "neutral", "bearish"):
            bias = "neutral"
        return bias, data.get("reasoning", "")
