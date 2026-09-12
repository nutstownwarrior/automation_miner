"""Models learned from this user's own decisions.

``ranking`` turns a person's accept/dismiss history into a calibrated estimate
of how likely they are to accept a *new* suggestion, used to order what they
see - never to decide what they see.  Later models in this package follow the
same rule: they may only touch presentation, not the backtest gate or the
conflict checker.
"""

from __future__ import annotations
