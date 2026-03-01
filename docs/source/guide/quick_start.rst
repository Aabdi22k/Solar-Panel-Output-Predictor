Quickstart
==========

This guide walks through installing and running the SPOP locally.

Prerequisites
-------------

- Python 3.10+
- pip
- Virtual environment recommended

Installation
------------

Clone the repository:

.. code-block:: bash

    git clone https://github.com/Aabdi22k/Solar-Panel-Output-Predictor-Remake.git
    cd Solar-Panel-Output-Predictor-Remake

Create and activate a virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows

Install dependencies:

.. code-block:: bash

    pip install -r requirements.txt

Run the Forecast App
--------------------

Launch the Streamlit interface:

.. code-block:: bash

    streamlit run src/main/app/app.py

The interface allows you to:

- Enter latitude and longitude
- Configure panel area and efficiency
- Generate a 5-day forecast
- View uncertainty bands

Project Structure
-----------------

::

    src/main/
        app/
        data_pipeline/
        data_sources/
        features/
        math/
        models/
        config.py
        paths.py
        schemas.py