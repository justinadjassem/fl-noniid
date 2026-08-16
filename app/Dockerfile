FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/srv

# Pas de torch ici : le dashboard n'appelle que l'API en HTTP.
COPY requirements/ /srv/requirements/
RUN pip install --no-cache-dir -r /srv/requirements/app.txt

COPY contracts/  /srv/contracts/
COPY .streamlit/ /srv/.streamlit/
COPY app/        /srv/app/

EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
