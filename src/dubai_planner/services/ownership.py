"""Who owns a session: visitor tokens and owner keys (deployment security).

A public Streamlit app has no accounts. Each browser tab's Streamlit session gets a
random **visitor token** (:func:`new_visitor_token`, 256 bits) that lives only in the
server-side ``st.session_state``: it is never sent to the browser, put in a URL, logged
or traced. Sessions store an **owner key**, the SHA-256 of that token, so the database
never holds a usable token either.

Identity is therefore *per browser tab*: a page refresh, a new tab or a server restart
starts a new Streamlit session and thus a new visitor, who cannot see the previous
visitor's sessions. If the app is configured for ``st.login`` (OIDC), the owner key is
derived from the authenticated identity instead and survives refreshes
(:meth:`Owner.from_identity`).

:attr:`Owner.LOCAL` (no key) is the single-user mode (``SESSION_SCOPE=shared``,
scripts, tests): it sees every session, including legacy ones that have no owner.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import ClassVar

TOKEN_BYTES = 32
#: ``secrets.token_urlsafe(32)`` yields 43 characters; anything shorter is rejected.
MIN_TOKEN_LENGTH = 43


class SessionNotFound(LookupError):
    """The session (or run) does not exist *or belongs to someone else*.

    Both cases raise the same error with the same message, so a visitor cannot tell a
    foreign session from a missing one.
    """

    def __init__(self) -> None:
        super().__init__("session not found")


def new_visitor_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def _key(domain: str, value: str) -> str:
    return hashlib.sha256(f"{domain}\n{value}".encode()).hexdigest()


@dataclass(frozen=True, repr=False)
class Owner:
    """The caller of the session services. ``key is None`` = local, unrestricted."""

    key: str | None = None

    LOCAL: ClassVar[Owner]

    @property
    def is_local(self) -> bool:
        return self.key is None

    @classmethod
    def from_token(cls, token: str) -> Owner:
        """A visitor, from the random token kept in its server-side Streamlit session."""
        if not isinstance(token, str) or len(token) < MIN_TOKEN_LENGTH:
            raise ValueError("invalid visitor token")
        return cls(_key("visitor", token))

    @classmethod
    def from_identity(cls, issuer: str, subject: str) -> Owner:
        """An authenticated user (OIDC ``iss`` + ``sub`` from ``st.user``)."""
        if not issuer or not subject:
            raise ValueError("incomplete identity")
        return cls(_key("oidc", f"{issuer}\n{subject}"))

    def __repr__(self) -> str:  # never print the key (it identifies a visitor's data)
        return "Owner(local)" if self.key is None else "Owner(visitor)"


Owner.LOCAL = Owner(None)
