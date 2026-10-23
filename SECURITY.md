# Security policy

## Supported versions

Security fixes land on `main` for the current development line of Lydlr.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Email the maintainers privately (repository owner: `jrb00013`) with:

- Affected component (backend API, ROS node, wire protocol, frontend)
- Reproduction steps on a system you are authorized to test
- Impact assessment (data exposure, RCE, auth bypass, etc.)

## Scope notes

- Lydlr compresses and transports sensor data. Treat fleet endpoints as sensitive.
- Keep API keys and Mongo credentials in `.env` (never commit).
- Wire protocol (`LYDT`) is not an authentication boundary — run it on trusted links or terminate TLS at the relay.
