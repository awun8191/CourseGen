from dataclasses import dataclass

@dataclass(frozen=True)
class RateLimit:
    per_day:   int
    per_minute: int

RATE_LIMITS = {
    "lite":  RateLimit(per_day=1000, per_minute=15),
    "flash": RateLimit(per_day=250, per_minute=10),
    "pro":   RateLimit(per_day=25, per_minute=5),
    "embedding": RateLimit(per_day=1000, per_minute=5)
}
