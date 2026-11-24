CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id        bigserial PRIMARY KEY,
    source    text NOT NULL,
    page      int NOT NULL,
    content   text NOT NULL,
    embedding vector(1536) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING gin (tsv);
CREATE INDEX ON chunks (source);

CREATE TABLE equipment (
    tag                 text PRIMARY KEY,
    model               text NOT NULL,
    kind                text NOT NULL CHECK (kind IN ('pump', 'transformer', 'compressor')),
    site                text NOT NULL,
    status              text NOT NULL CHECK (status IN ('running', 'standby', 'maintenance')),
    installed_on        date NOT NULL,
    last_service        date NOT NULL,
    hours_since_service int NOT NULL CHECK (hours_since_service >= 0)
);

INSERT INTO equipment VALUES
    ('P-101', 'HX-200', 'pump', 'Cairo', 'running', '2021-03-14', current_date - 400, 5200),
    ('P-102', 'HX-200', 'pump', 'Cairo', 'running', '2021-03-14', current_date - 120, 2100),
    ('P-103', 'HX-200', 'pump', 'Cairo', 'standby', '2022-06-02', current_date - 300, 900),
    ('P-104', 'HX-200', 'pump', 'Alexandria', 'running', '2022-06-02', current_date - 200, 4300),
    ('P-105', 'HX-200', 'pump', 'Alexandria', 'maintenance', '2023-01-20', current_date - 10, 0),
    ('P-106', 'HX-200', 'pump', 'Alexandria', 'running', '2023-01-20', current_date - 90, 1500),
    ('T-201', 'TR-500', 'transformer', 'Cairo', 'running', '2020-11-05', current_date - 380, 9100),
    ('T-202', 'TR-500', 'transformer', 'Cairo', 'running', '2020-11-05', current_date - 150, 3600),
    ('T-203', 'TR-500', 'transformer', 'Alexandria', 'running', '2022-02-17', current_date - 330, 7900),
    ('C-301', 'AC-75', 'compressor', 'Cairo', 'running', '2021-09-09', current_date - 160, 3800),
    ('C-302', 'AC-75', 'compressor', 'Cairo', 'standby', '2021-09-09', current_date - 60, 650),
    ('C-303', 'AC-75', 'compressor', 'Alexandria', 'running', '2023-05-30', current_date - 220, 4150),
    ('C-304', 'AC-75', 'compressor', 'Alexandria', 'running', '2023-05-30', current_date - 30, 700);

CREATE TABLE work_orders (
    id            bigserial PRIMARY KEY,
    equipment_tag text NOT NULL REFERENCES equipment (tag),
    priority      text NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    summary       text NOT NULL CHECK (length(summary) BETWEEN 5 AND 500),
    status        text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'done')),
    thread_id     text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX ON work_orders (equipment_tag, status);

CREATE ROLE agent_reader LOGIN PASSWORD 'agent_reader';
ALTER ROLE agent_reader SET default_transaction_read_only = on;
ALTER ROLE agent_reader SET statement_timeout = '3s';
GRANT SELECT ON equipment, work_orders TO agent_reader;
