"""Robot and wall assets of the PHYBot C2 tasks.

mjlab is imported first: importing it loads its task plugins, these tasks
among them, and they import this package's modules. Loading mjlab before
any of those modules starts keeps that from re-entering a half-imported
module, which mjlab would only report as a warning before dropping the
tasks.
"""

import mjlab  # noqa: F401
