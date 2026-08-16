FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/srv

# Sur Linux, `pip install torch` tire la build CUDA (~2,5 Go) dont on n'a
# aucun usage : tout tourne sur CPU. On installe la build CPU depuis l'index
# dedie AVANT le reste ; pip la considerera ensuite comme satisfaite.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.10.0 torchvision==0.25.0

COPY requirements/ /srv/requirements/
RUN pip install --no-cache-dir -r /srv/requirements/api.txt

COPY contracts/ /srv/contracts/
COPY fl_core/   /srv/fl_core/
COPY api/       /srv/api/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
