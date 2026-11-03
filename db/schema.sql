-- Production data layer for the AWS deployment target (PostgreSQL 16 + PostGIS on RDS, private
-- subnet, KMS-encrypted). The Modal deployment in this repo keeps the same Job record in a
-- modal.Dict behind the JobStore interface (soulhome/service/store.py); this schema is what that
-- interface maps onto in Postgres. Managed with Alembic in production; plain DDL here for review.

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TYPE scan_tier   AS ENUM ('lidar', 'arcore', 'panorama');
CREATE TYPE scan_status AS ENUM ('queued', 'running', 'retrying', 'succeeded', 'failed');

CREATE TABLE scans (
    id               text PRIMARY KEY,                  -- scn_<16 hex>
    user_id          uuid        NOT NULL,
    tier             scan_tier   NOT NULL,
    status           scan_status NOT NULL DEFAULT 'queued',
    meta             jsonb       NOT NULL DEFAULT '{}', -- ScanMeta: camera height, north, units...
    input_prefix     text        NOT NULL,              -- s3://bucket/scans/<id>/input/
    idempotency_key  text,
    request_hash     char(64),
    attempts         smallint    NOT NULL DEFAULT 0,
    max_attempts     smallint    NOT NULL DEFAULT 3,
    current_stage    text,
    error            jsonb,                             -- {stage, code, message, retryable}
    dead_lettered    boolean     NOT NULL DEFAULT false,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    started_at       timestamptz,
    finished_at      timestamptz,
    UNIQUE (user_id, idempotency_key)                   -- atomic idempotency (INSERT ... ON CONFLICT)
);
CREATE INDEX scans_active_idx ON scans (status) WHERE status IN ('queued', 'running', 'retrying');
CREATE INDEX scans_failed_stage_idx ON scans ((error->>'stage'), (error->>'code')) WHERE status = 'failed';

-- One row per (scan, pipeline version): re-processing never overwrites earlier results.
CREATE TABLE scan_results (
    scan_id          text REFERENCES scans(id) ON DELETE CASCADE,
    pipeline_version text        NOT NULL,
    schema_version   text        NOT NULL,              -- spatial contract semver
    model            jsonb       NOT NULL,              -- full SpatialModel document
    artifact_prefix  text        NOT NULL,              -- s3://bucket/scans/<id>/results/v<ver>/
    confidence       real        NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (scan_id, pipeline_version)
);

CREATE TABLE stage_runs (                               -- per-stage observability
    scan_id     text REFERENCES scans(id) ON DELETE CASCADE,
    attempt     smallint NOT NULL,
    stage       text     NOT NULL,
    status      text     NOT NULL,                      -- ok | skipped | failed
    duration_ms real     NOT NULL,
    detail      jsonb,
    PRIMARY KEY (scan_id, attempt, stage)
);

-- Spatial tables in a local metric frame (SRID 0, metres) so the rule engine can ask geometric
-- questions in SQL ("objects within 0.5 m of the NE corner", "doors on the north-facing wall").
CREATE TABLE rooms (
    scan_id          text NOT NULL,
    pipeline_version text NOT NULL,
    room_id          text NOT NULL,
    name             text,
    footprint        geometry(Polygon, 0) NOT NULL,
    ceiling_height_m real NOT NULL,
    north_deg        real,
    PRIMARY KEY (scan_id, pipeline_version, room_id),
    FOREIGN KEY (scan_id, pipeline_version) REFERENCES scan_results ON DELETE CASCADE
);
CREATE INDEX rooms_footprint_gix ON rooms USING gist (footprint);

CREATE TABLE walls (
    scan_id text, pipeline_version text, room_id text, wall_id text,
    geom       geometry(LineString, 0) NOT NULL,
    height_m   real NOT NULL,
    facing_deg real,
    PRIMARY KEY (scan_id, pipeline_version, room_id, wall_id),
    FOREIGN KEY (scan_id, pipeline_version, room_id) REFERENCES rooms ON DELETE CASCADE
);

CREATE TABLE openings (
    scan_id text, pipeline_version text, room_id text, opening_id text,
    wall_id       text NOT NULL,
    kind          text NOT NULL CHECK (kind IN ('door', 'window', 'opening')),
    geom          geometry(LineString, 0) NOT NULL,     -- span along the wall
    sill_height_m real NOT NULL,
    height_m      real NOT NULL,
    PRIMARY KEY (scan_id, pipeline_version, room_id, opening_id),
    FOREIGN KEY (scan_id, pipeline_version, room_id) REFERENCES rooms ON DELETE CASCADE
);

CREATE TABLE objects (
    scan_id text, pipeline_version text, room_id text, object_id text,
    catalog_id   text NOT NULL,
    confidence   real NOT NULL,
    position     geometry(PointZ, 0) NOT NULL,
    direction_16 text,
    grid_zone    text,
    PRIMARY KEY (scan_id, pipeline_version, room_id, object_id),
    FOREIGN KEY (scan_id, pipeline_version, room_id) REFERENCES rooms ON DELETE CASCADE
);
CREATE INDEX objects_position_gix ON objects USING gist (position);

-- Example rule-engine query: fire-element objects in the north-east zone.
-- SELECT * FROM objects WHERE grid_zone = 'NE' AND catalog_id IN ('stove', 'oven', 'diya');
