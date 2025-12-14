# NYC Collision Data ETL Pipeline

A  ETL pipeline designed to analyze New York City traffic collision data. It ingests raw data, enriches it with weather and holiday context and aggregates insights for policy-making.

Built with **Python**, **Polars**, and **DuckDB** following the **Medallion Architecture** (Bronze → Silver → Gold).

---

## Architecture

* **Bronze Layer:** Raw ingestion from NYC Open Data, Nager.Date (Holidays) and NOAA (Weather).
* **Silver Layer:** Data cleaning, standardization and partitioning (Parquet).
* **Gold Layer:** Business logic, enrichment (joins) and final aggregation for reporting.

---

## Prerequisites

* **Python 3.11+**
* **Poetry** (Dependency Manager)

If you don't have Poetry installed, follow the [official instructions](https://python-poetry.org/docs/#installation).

---

## Installation

This project uses **Poetry** to manage dependencies and virtual environments strictly.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/antonioRubi/uptimal-use-case.git](https://github.com/antonioRubi/uptimal-use-case.git)
    cd uptimal-use-case
    ```

2.  **Install dependencies:**
    ```bash
    poetry install
    ```
    *This will create a virtual environment and install all required libraries (Polars, DuckDB, Pytest, etc.).*

---

## How to Run

You can execute the entire pipeline with a single command:

```bash
poetry run python main.py
```

## AI Usage

In alignment with current engineering practices and efficiency, this project utilized Generative AI as a development accelerator.

* **Human Ownership (Architecture & Logic):**
    * Designed the **Medallion Architecture** (Bronze/Silver/Gold) and data flow.
    * Defined the **Business Logic** for holiday categorization and weather impact.
    * Selected the technology stack (**Polars**, **DuckDB**, **Poetry**).

* **AI Assistance (Implementation & Polish):**
    * **Refactoring:** Used to ensure code consistency and type hinting across modules.
    * **Testing:** Generated `pytest` scaffolding and mock data fixtures to ensure robust test coverage.
    * **Documentation:** Assisted in generating clear docstrings and comments.
    * **Visualization:** Generated the `matplotlib`/`seaborn` syntax for the analysis notebook to speed up the reporting phase as I don't have that much experience with plotting on Python