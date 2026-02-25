"""Cost tracking utility for API calls.

Logs API calls and estimates costs for different services.
Provides a simple interface to track usage and print summaries.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class APICall:
    """Record of a single API call."""
    service: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class CostTracker:
    """Tracks API calls and estimates costs.
    Supports multiple services with hardcoded cost estimates (per 1M tokens, USD):
    - Gemini 2.5 Flash:      $0.30 input (text/image) / $2.50 output (stable, since ~Jun 2025)
    - Gemini 2.5 Flash-Lite:  $0.10 input / $0.40 output (budget variant for throughput-heavy tasks)
    - OpenAI Realtime:        $0.06 per minute of audio (approximate)
    
    Note: Gemini batch mode can roughly halve costs (~$0.15 in / $1.25 out for 2.5 Flash).
    """

    # Cost estimates per token (USD) - Updated February 2026
    COSTS = {
        "gemini": {
            "gemini-2.5-flash": {
                "input_tokens": 0.30 / 1_000_000,   # $0.30 per 1M input
                "output_tokens": 2.50 / 1_000_000,  # $2.50 per 1M output
            },
            "gemini-2.5-flash-lite": {
                "input_tokens": 0.10 / 1_000_000,   # $0.10 per 1M input
                "output_tokens": 0.40 / 1_000_000,  # $0.40 per 1M output
            },
        },
        "openai": {
            "gpt-realtime": {
                "input_tokens": 32.00 / 1_000_000,    # $5.00 per 1M text tokens
                "output_tokens": 64.00 / 1_000_000,  # $20.00 per 1M text tokens
                "audio_input_tokens": 40.00 / 1_000_000, # $40.00 per 1M audio tokens
                "audio_output_tokens": 80.00 / 1_000_000, # $80.00 per 1M audio tokens
                "audio_minute": 0.06,  # Approx $0.06 per minute
            },
            "gpt-realtime-mini": {
                "input_tokens": 0.60 / 1_000_000,    # $0.60 per 1M text tokens
                "output_tokens": 2.40 / 1_000_000,   # $2.40 per 1M text tokens
                "audio_input_tokens": 10.00 / 1_000_000, # $10.00 per 1M audio tokens
                "audio_output_tokens": 20.00 / 1_000_000, # $20.00 per 1M audio tokens
                "audio_minute": 0.02,  # Approx $0.02 per minute
            },
        },
    }

    def __init__(self) -> None:
        """Initialize the cost tracker."""
        self._calls: list[APICall] = []
        self._session_start = datetime.now()

    def log_call(
        self,
        service: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_seconds: float = 0.0,
    ) -> None:
        """Log an API call with metadata.
        
        Args:
            service: Service name (e.g., 'gemini', 'openai')
            model: Model name (e.g., 'gemini-2.5-flash')
            input_tokens: Number of input tokens (for LLM calls)
            output_tokens: Number of output tokens (for LLM calls)
            duration_seconds: Duration of the call in seconds (for audio)
        """
        call = APICall(
            service=service,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_seconds=duration_seconds,
        )
        self._calls.append(call)
        logger.debug(
            "Logged API call: %s/%s (in=%d, out=%d, duration=%.1fs)",
            service,
            model,
            input_tokens,
            output_tokens,
            duration_seconds,
        )

    def get_session_cost(self) -> dict:
        """Return cost breakdown by service.
        
        Returns:
            Dict with structure:
            {
                "total_cost": float,
                "by_service": {
                    "gemini": {"calls": int, "cost": float, "details": {...}},
                    "openai": {"calls": int, "cost": float, "details": {...}},
                },
                "call_count": int,
            }
        """
        by_service: dict = {}
        total_cost = 0.0

        for call in self._calls:
            if call.service not in by_service:
                by_service[call.service] = {
                    "calls": 0,
                    "cost": 0.0,
                    "details": {},
                }

            by_service[call.service]["calls"] += 1

            # Calculate cost for this call
            call_cost = self._estimate_call_cost(call)
            by_service[call.service]["cost"] += call_cost
            total_cost += call_cost

            # Track details by model
            if call.model not in by_service[call.service]["details"]:
                by_service[call.service]["details"][call.model] = {
                    "calls": 0,
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "duration_seconds": 0.0,
                }

            model_detail = by_service[call.service]["details"][call.model]
            model_detail["calls"] += 1
            model_detail["cost"] += call_cost
            model_detail["input_tokens"] += call.input_tokens
            model_detail["output_tokens"] += call.output_tokens
            model_detail["duration_seconds"] += call.duration_seconds

        return {
            "total_cost": round(total_cost, 4),
            "by_service": by_service,
            "call_count": len(self._calls),
        }

    def print_summary(self) -> None:
        """Log a formatted cost summary."""
        cost_data = self.get_session_cost()
        
        summary_lines = []
        summary_lines.append("\n" + "=" * 70)
        summary_lines.append("COST SUMMARY")
        summary_lines.append("=" * 70)
        
        if cost_data["call_count"] == 0:
            summary_lines.append("No API calls logged.")
            summary_lines.append("=" * 70 + "\n")
            logger.info("\n".join(summary_lines))
            return

        summary_lines.append(f"Total API calls: {cost_data['call_count']}")
        summary_lines.append(f"Total estimated cost: ${cost_data['total_cost']:.4f}")
        summary_lines.append("")

        for service, service_data in cost_data["by_service"].items():
            summary_lines.append(f"  {service.upper()}")
            summary_lines.append(f"    Calls: {service_data['calls']}")
            summary_lines.append(f"    Cost: ${service_data['cost']:.4f}")
            
            for model, model_detail in service_data["details"].items():
                summary_lines.append(f"      {model}")
                summary_lines.append(f"        Calls: {model_detail['calls']}")
                summary_lines.append(f"        Cost: ${model_detail['cost']:.4f}")
                
                if model_detail["input_tokens"] > 0 or model_detail["output_tokens"] > 0:
                    summary_lines.append(f"        Tokens: {model_detail['input_tokens']} in, {model_detail['output_tokens']} out")
                
                if model_detail["duration_seconds"] > 0:
                    summary_lines.append(f"        Duration: {model_detail['duration_seconds']:.1f}s")
            summary_lines.append("")

        summary_lines.append("=" * 70 + "\n")
        logger.info("\n".join(summary_lines))

    def _estimate_call_cost(self, call: APICall) -> float:
        """Estimate the cost of a single API call.
        
        Args:
            call: APICall record
            
        Returns:
            Estimated cost in USD
        """
        service_costs = self.COSTS.get(call.service, {})
        model_costs = service_costs.get(call.model, {})

        cost = 0.0

        # Token-based pricing (Gemini)
        if "input_tokens" in model_costs and call.input_tokens > 0:
            cost += call.input_tokens * model_costs["input_tokens"]
        if "output_tokens" in model_costs and call.output_tokens > 0:
            cost += call.output_tokens * model_costs["output_tokens"]

        # Duration-based pricing (OpenAI Realtime)
        if "audio_minute" in model_costs and call.duration_seconds > 0:
            minutes = call.duration_seconds / 60.0
            cost += minutes * model_costs["audio_minute"]

        return cost
