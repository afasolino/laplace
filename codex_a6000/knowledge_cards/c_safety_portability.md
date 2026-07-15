# Project-local C safety precedence card

Repository conventions and public headers define the interface. Check ownership,
sizes, integer ranges, partial I/O, and cleanup on every negative path. C11 code
must pass warnings-as-errors plus AddressSanitizer and UndefinedBehaviorSanitizer
tests where supported; compiler success alone is not approval.
