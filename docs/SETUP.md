# Setup

Install Python 3.14+, uv, PostgreSQL, and Ollama. Clone the repo and run `uv sync --frozen`. The reference backend is PostgreSQL; no Docker deployment is required.

## Database

In an administrative psql session, create a dedicated local role and databases. Set your own password interactively:

```sql
CREATE ROLE pcr_app LOGIN;
\password pcr_app
CREATE DATABASE pcr OWNER pcr_app;
CREATE DATABASE pcr_test OWNER pcr_app;
```

On Linux an installed server may expose that session through `sudo -u postgres psql`; on Windows use SQL Shell (psql) or your configured administrative client.

PowerShell: `Copy-Item .env.example .env`. Linux/macOS: `cp .env.example .env`.

Edit the connection URLs with your role and password. URL-encode special characters in credentials. Apply `schema.sql` to the runtime database using its actual connection URL:

```sh
psql "postgresql://pcr_app:YOUR_PASSWORD@localhost:5432/pcr" -f schema.sql
```

The test harness applies the schema automatically to its disposable database.

## Model and storage

Start Ollama, pull `qwen3:4b-instruct-2507-q4_K_M`, and run `uv run pcr`. Keep the same model when comparing behavior with the source project. Resource admission may reject a model that does not fit available host resources.

Use a dedicated artifact root. The inherited environment name is `PROMETHEIST_ARTIFACT_ROOT`; `.env.example` uses the checkout's ignored `.prometheist/artifacts` folder. Do not point this extraction at the live application's database or artifact root.

Local loopback model/database endpoints work without remote-service enrollment. Nonlocal destinations retain the source engine's explicit outbound-consent policy. No hosted model service is required.

## Tests

Set the test URL in the process environment; the test harness chooses it before application `.env` loading.

```powershell
$env:TEST_DATABASE_URL = "postgresql://pcr_app:YOUR_PASSWORD@localhost:5432/pcr_test"
uv run pytest -q -ra
```

```sh
export TEST_DATABASE_URL='postgresql://pcr_app:YOUR_PASSWORD@localhost:5432/pcr_test'
uv run pytest -q -ra
```

Only use a disposable database: its tables are cleared before tests. See [TESTING.md](TESTING.md).
