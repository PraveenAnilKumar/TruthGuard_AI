# TruthGuard AI Current Table Design

This document mirrors the `tbl_login` sample format, but it is based only on the data structures that are implemented in the current project.

TruthGuard AI does not currently use a relational database schema. The only persisted application data store verified in the codebase is `users.json`, which is managed by the authentication helpers in `app.py`.

Implementation note: the underlying store is a JSON object keyed by `username`. The table below is a documentation view of that structure in table-design form.

Admin credentials are handled separately through environment variables (`ADMIN_USER` and `ADMIN_PASSWORD`) and are not stored as rows in `users.json`.

## 1. tbl_users

Primary key: `username`

| No | Fieldname | Datatype (Size) | Key Constraints | Description of the Field |
| --- | --- | --- | --- | --- |
| 1 | username | string (variable length, minimum 3 characters enforced) | Primary Key, NOT NULL, UNIQUE | Username used as the JSON key and as the login identifier. |
| 2 | password | string (60-64 characters stored) | NOT NULL | Password hash for the user. Current registrations use bcrypt hashes; legacy SHA-256 hashes may still appear until the user logs in and the hash is upgraded. |
| 3 | role | string (current values: `user`, `admin`) | NOT NULL | Role assigned to the user account. |
| 4 | created_at | datetime string (ISO 8601, `YYYY-MM-DDTHH:MM:SS`) | NULL allowed for legacy records | Account creation timestamp written for newly registered users. Older records may not have this field. |

## Notes Against the Reference Format

- There is no persisted `login_id` field in the current project.
- There is no persisted `email` field in the current project.
- There is no persisted numeric `type` field; role is stored as a string in `role`.
- There is no persisted numeric `status` field for active or blocked users.
- The admin account is environment-based and should not be documented as a persisted table row.

## Non-Table Storage in the Current Project

The following project storage locations are implementation artifacts, not business tables:

- `models/`
- `temp/`
- `datasets/`

## Verification Basis

This table design is based on the current implementation in:

- `app.py` authentication helpers: `load_users()`, `save_users()`, `register_user()`
- `app.py` login flow for local users and environment-based admin login
- `users.json` sample persisted records
- `docs/truthguard_ai_sequence_diagram.md`, which also describes `users.json` as the auth data store
