# C11 portability and safety card

Release: `multilanguage-corpus-v1`. Licence: MIT. This is an independently
authored Laplace reference card; it does not reproduce a proprietary language
standard.

## Interfaces and ownership

- State whether each pointer is borrowed, transferred, or returned to the
  caller. Define which function releases every acquired resource.
- Validate pointer and length pairs before dereference. Treat a null pointer as
  valid only when the public contract explicitly permits it for a zero length.
- Acquire resources in a visible order and release them in reverse order on
  every error path. Preserve the primary error before cleanup can overwrite it.
- Never return a pointer to an automatic object. Do not use a pointer after
  `free`, `fclose`, or a successful ownership transfer.

## Integers, buffers, and undefined behavior

- Use `size_t` for object sizes and indexes. Check `count > SIZE_MAX / width`
  before multiplication and check `a > SIZE_MAX - b` before addition.
- Check range before narrowing or converting a signed value to an unsigned
  type. Avoid shifting a negative value or shifting by a count greater than or
  equal to the promoted operand width.
- Define whether a bounded text output includes its terminating null byte.
  Reject insufficient capacity without writing a partial success result unless
  partial output is part of the interface contract.
- Use `memmove` for potentially overlapping objects. Do not infer that a short
  read, short write, or interrupted system call is a complete operation.

## Errors, files, and processes

- Return one documented success value and explicit error values. When `errno`
  is part of the interface, inspect it only after a function reports failure.
- Check `ferror` independently from end-of-file. Verify `fflush`, `fclose`, and
  final rename results when durability or atomic replacement is required.
- Treat process exit status, signal termination, and spawn/exec failure as
  distinct outcomes. Do not invoke a shell for data that originated outside a
  trusted static configuration.

## Deterministic gates

Compile as C11 with `-Wall -Wextra -Wpedantic -Werror`, run self-checking unit
tests, then repeat the tests with AddressSanitizer and UndefinedBehaviorSanitizer
where supported. A warning-free compile is not evidence of functional safety;
tests must exercise invalid lengths, allocation failure boundaries, empty
input, maximum values, partial I/O, and cleanup after failure.
