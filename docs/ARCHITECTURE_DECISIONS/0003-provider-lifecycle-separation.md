# ADR 0003: separate invocation from lifecycle

Status: accepted

A provider invokes a configured local endpoint. It never downloads a model or
controls a process. Model process start/stop remains in the existing lifecycle
service and requires a matching ownership record. Unowned endpoints can never be
stopped by Laplace.
