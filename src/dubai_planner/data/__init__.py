"""Data layer: cached HTTP client, public-source clients and curated seed data.

Clients: ``open_meteo`` (weather), ``overpass`` (OSM POIs), ``nominatim`` (geocoding),
``wikivoyage`` (descriptions), ``fx`` (Frankfurter + AED peg), ``calendar`` (Ramadan,
holidays, events). All are keyless, cached, and fall back to ``seed`` when offline.
"""
